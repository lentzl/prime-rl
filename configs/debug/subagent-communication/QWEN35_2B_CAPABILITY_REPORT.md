# Qwen3.5 2B Capability Report

This report freezes the conclusions from the Qwen3.5 2B Prime Agent curriculum before
the coordinator experiment moves to Qwen3.5 4B. It distinguishes a durable model
artifact from experimental checkpoints and distinguishes demonstrated skills from
capabilities that remain prompt-sensitive.

## Retained checkpoint

The selected 2B checkpoint remains rung 37 step 2. It is a LoRA adapter trained over
the retained causal-conjunction base and merged into a standalone dense model for
portable inference and future training.

| Item | Value |
| --- | --- |
| HF repository | `lentzl/rlm-prime-agent-qwen35-orchestrator-candidate-r1-20260809` |
| Pinned revision | `b469454738dfc911f43233f172ca4ff920ea695d` |
| Visibility | private |
| Durable format | complete dense BF16 model, not an adapter-only dependency chain |
| Remote dense source | `/ephemeral/subagent-rung/outputs/68-rung37-dense-export-r1/weights/step_1` |
| Selected adapter source | `/ephemeral/subagent-rung/outputs/37-single-path-opsd-dose-r1/weights/step_2` |
| EOS contract | `<|im_end|>` = `248046` in tokenizer and every numeric model field |

The HF snapshot contains the model, tokenizer, chat template, generation metadata,
training configs, source-state manifests, and a `STABLE` marker. The model repository
is the durable source; paths on a rented machine are caches and may disappear.

The dense export was checked tensor by tensor. All 96 adapted matrices matched the
LoRA merge equation, all 521 untouched tensors were bit-identical to the parent, and
the adapter and dense model produced the same greedy spawn action. The snapshot is
therefore a faithful export of the selected policy.

## Demonstrated capabilities

The 2B model has a useful Prime Agent substrate and remains a credible starting point
for a small expert child:

- It accepts silent IPython assignments and reuses state across later cells and user
  turns, including after the original source file has disappeared.
- It can stop cleanly after a direct computation without leaking ChatML or synthetic
  tool turns.
- It performs ordinary file discovery, download, structured-result inspection, and
  common CSV and PDF extraction substantially better than the untouched base.
- It reliably avoids delegation on the direct arithmetic gate (`4/4` on repeated
  standard-prompt comparisons).
- It can spawn a named child with a concrete path and solve single-child tasks. The
  guided gate reached `3/3`; standard gates varied with seed.
- It can fan out to two children, retain handles, receive replies, and synthesize an
  exact result in successful trajectories. The guided parallel gate reached `3/3`
  exact answers and `2/3` fully aligned provenance.
- It can execute the child-to-parent request primitive reliably when the child prompt
  names the exact `agent_message.send` API. In a direct child-role probe, all `24/24`
  guided follow-up samples sent the request after computing the subtotal.

These are learned capabilities, not complete harness mastery. Across disjoint
12-task standard direct/single/parallel gates, rung 37 repeatedly preserved direct
work but varied between adjacent delegated families. The three gates retained in the
frozen capacity battery scored `4/4, 3/4, 3/4`, `4/4, 4/4, 2/4`, and
`4/4, 4/4, 3/4`, respectively. Other paired experiment gates were as low as `8/12`
and `9/12` under different held-out seeds. The frozen best was therefore `11/12`, but
no split established simultaneous `4/4` mastery of single and parallel work.

## Outstanding problems

The selected 2B checkpoint is not admitted as the general coordinator:

- Standard-language coordinator prompts are unstable. Spawn payloads can omit the
  path, reverse parent and child responsibilities, leak withheld values, or include
  invalid arguments.
- Parallel fan-in and single-child completion interfere. Narrow GRPO, OPD, OPSD, and
  mixed SDPO updates repeatedly moved successes from one family to the other instead
  of improving both.
- Bidirectional communication is not learned under natural instructions. No rung
  completed the causal `child request -> parent response -> child result` chain on a
  held-out standard-prompt screen.
- The child does not robustly translate ordinary language such as "send the subtotal
  to your parent" into `await agent_message.send(...)`. It often emits plain assistant
  text even after a successful compute cell.
- The coordinator still invents polling or messaging operations, loses handles, or
  continues after enough evidence is available in difficult samples.
- Traceback repair, strict output contracts, semantic faithfulness, and bounded
  no-repeat behavior remain uneven outside the focused foundation gates.

The strongest capacity diagnostic is teacher admission, not a failed optimizer run.
The frozen 2B teacher produced the exact child request in `24/24` guided post-compute
samples, but only `0/16` standard samples with the original task demonstration. A
response-aligned, request-only demonstration still produced a real tool call in only
`1/24` samples; the other 23 emitted plain text. In this context the teacher cannot
supply a reliable SDFT target for the missing standard-language transition.

## Experiments that did not supersede rung 37

- Response-filtered and complete-fan-in OPSD could move the intended behavior, but
  traded single-child and parallel reliability.
- Process-control GRPO used a valid diagnostic reward but trained mostly saturated
  examples and regressed the held-out gate from `9/12` to `7/12`.
- Hard-example GRPO and frozen-policy OPD moved the single/parallel split in opposite
  directions without increasing joint mastery.
- Branch-matched full-weight SDPO was stable end to end but scored `9/12`.
- Mixed full-weight SDPO plus OPSD retention reached a balanced `10/12`, but did not
  dominate rung 37 on the paired family split.
- Four full-weight guided child-request doses produced no aligned bidirectional
  result. The response-phase repair rung also scored `0/8` on its held-out
  bidirectional screen.

These failures are retained as causal evidence. They are not candidates to upload or
use as expert bases.

## Intended reuse

Keep the dense rung-37 snapshot indefinitely as:

1. the smallest demonstrated Prime Agent and IPython-capable policy;
2. the initialization for a 2B expert child, where narrow specialization and typed
   outputs require less global coordination capacity;
3. a frozen baseline for measuring whether a 4B coordinator actually buys protocol
   reliability rather than merely more parameters;
4. a possible low-cost worker in a heterogeneous recursive system.

Do not call it the RLM Master and do not overwrite its HF repository with later failed
checkpoints. Any expert specialization should publish to a new repository so the
general 2B substrate remains reproducible.
