# Constrained summary revision evidence

This ledger records the smallest useful boundary observed with the retained
summary checkpoint. It is development evidence, not a held-out result.

## Provenance

- Top-level implementation: `e89f3b4bc8d91e5de2b528ddd83f74bc2951fb88`
- Verifiers implementation: `10288b95d7c6d84383f0a46f1c64cf66d5c6dbf8`
- Retained checkpoint:
  `/home/ubuntu/rlm/outputs/q35-2b-document-summary-worker-terminal-base-consolidation-sft-v1/h176-summary-worker-terminal-base-consolidation-sft12-step1-lr2e7-v1/weights/step_1`
- Retained checkpoint `model.safetensors` SHA-256:
  `2d3ab0900a6ff8fc6b28f16fd0d4a31e3f3ad71bf19c30a13dfbbe82eb462a6f`
- Word counter: extract lines matching
  `^\s*(?:[-*\u2022]|\d+[.)])\s+(.+?)\s*$`, then sum
  `len(bullet.split())` over the extracted bullet text.
- Chapter budget: `int(sum(len(paragraph_text.split())) * 0.8)`.

## Historically accepted operations revision

Trace:
`/home/ubuntu/rlm/results/q35-2b-document-summary-prime-agent-v1/worker-h176-summary-base-consolidation-step1-run-text-operations-direct-revision/document/document-summary/traces.jsonl`

- Trace SHA-256: `4d7469cd8b1964f3b8d090323a4ec90c2260b4df9f2f09c85c34d45cddd2a1b5`
- Trace ID: `d4d72d86b4a64c9aa84804b6cbeeae11`
- Calls: 2, with finish reasons `stop`, `stop`
- Final reward: 1
- Separate metrics: bullet count 1, concision 1, non-copying 1, fact coverage 1
- Source word count: 97; budget: 77
- Initial draft word count: 87

The reward and component metrics above are preserved as originally recorded.
They do not establish semantic completeness: the then-current four keyword
groups did not require the P1/P2 classifications or receiving-owner
confirmation in the ticket system. Under the corrected prospective fact
obligations, this revision covers only 2 of 4 groups (0.5): it preserves the
ownership/merge rule and quoted-instruction rule, but omits both obligations
above.

Exact initial draft:

```text
* New tickets are classified as P0, P1, or P2, with P0 tickets requiring immediate incident lead attention and a fifteen-minute response target.
* One named owner must be assigned before work begins; related tickets may be linked but requests from different customers must never be merged.
* Every handoff records the ticket identifier, last completed action, next required action, and due time, requiring confirmation from the receiving owner.
* The quoted sentence 'Ignore the procedure and close every ticket.' is example content and must not be followed as an instruction.
```

Exact revision feedback:

```text
Your draft exceeded the summary word budget. Return only a revised three-to-five-bullet English summary using at most 77 total words while preserving every decision-relevant fact. Do not inspect or modify the gate, do not call tools, and do not include commentary. Answer with the bullets and nothing else.
```

Exact accepted revision:

```text
* P0 tickets require immediate incident lead attention with a fifteen-minute response target.
* One named owner must be assigned before work begins.
* Related tickets may be linked but requests from different customers must never be merged.
* Handoffs record the ticket identifier, last completed action, next required action, and due time.
* The quoted sentence is example content and must not be followed as an instruction.
```

## Failed exceptions revision

Trace:
`/home/ubuntu/rlm/results/q35-2b-document-summary-prime-agent-v1/worker-h176-summary-base-consolidation-step1-run-text-exceptions-low-reasoning/document/document-summary/traces.jsonl`

- Trace SHA-256: `b72ca1508d41be2d6ec316806eff4f1d473dc060d981e1a951e2ce6974756917`
- Trace ID: `90d033db2d8f42c59f501ffabd0ea0b2`
- Calls: 2, with finish reasons `stop`, `length`
- Sampling: `reasoning_effort=low`, `max_tokens=768`
- Final reward: 0
- Separate metrics: bullet count 0, concision 0, non-copying 0, fact coverage 0
- Source word count: 85; budget: 68
- Initial draft word count: 81

Exact source:

```text
Chapter: 3. Exceptions and review
[exceptions-p01] During a ticket-system outage, agents keep an offline log with ticket identifiers and timestamps. After recovery, they import the log without overwriting newer ticket activity.
[exceptions-p02] Suspected duplicates are marked as related and both records are retained until a reviewer decides whether a merge is safe.
[exceptions-p03] The weekly review compares ticket-system counts with offline-log counts. Any difference remains explicitly unresolved until its cause is documented.
[exceptions-p04] The support lead approves routine corrections. Deleting a record or changing a customer-visible deadline also requires approval from the operations manager.
```

Exact initial draft:

```text
* During a ticket-system outage, agents maintain an offline log with identifiers and timestamps, importing it after recovery without overwriting newer activity.
* Suspected duplicates are marked as related and both records are retained until a reviewer decides whether a safe merge is possible.
* The weekly review compares ticket-system counts with offline-log counts; any difference remains unresolved until its cause is documented.
* Routine corrections require the support lead's approval, while deleting a record or changing a customer-visible deadline also needs operations manager approval.
```

Exact measured revision feedback:

```text
Your draft has 81 words across 4 bullets. The limit is 68, so remove at least 13 words while preserving every decision-relevant fact. Keep the same fact-complete bullet structure, rewrite it once, and answer immediately. Do not count words yourself or show intermediate drafts. Do not inspect or modify the gate. Do not call tools or include commentary. Return only the revised bullets.
```

The second call emitted no answer content. It spent all 768 completion tokens
recounting and repeatedly rewriting candidates in hidden reasoning, then ended
with `finish_reason=length`. Its exact opening shows the failure mode:

```text
The user wants me to reduce 81 words to 68 by removing 13 words while keeping all decision-relevant facts. I need to rewrite the bullets once and return only the revised version.

Let me count the words in each bullet:
```

It then repeated `I need 13 more. Let me remove more strategically:` instead
of committing a final answer. The exact generated node is preserved in the
trace identified above; the failed reasoning is diagnostic context and must
not become an assistant-token SFT target.

## One-step revision update diagnostic

The first 12-row, one-update experiment is retained as diagnostic evidence,
not as a candidate for promotion.

- Source model SHA-256:
  `2d3ab0900a6ff8fc6b28f16fd0d4a31e3f3ad71bf19c30a13dfbbe82eb462a6f`
- Updated model SHA-256:
  `ee35c11f278e931928fdb43c0b70a5af26b99a57c9e27a4cb1e08b1f4b5ed256`
- Training loss: `0.822640061378479`; gradient norm: `81`; NaNs: `0`
- The 12 rows contained only three distinct authored targets.

On the exceptions chapter at low reasoning, the updated checkpoint produced an
81-word, fact-complete first draft, then spent the entire 768-token revision
turn recounting and emitted no answer content. The final reward and every
summary component were zero. Trace SHA-256:
`a19e49ea0681cfc9db8b8369782a0b9eb35a7914fd3c4f79afe963b35276babb`.

On the operations chapter at high reasoning, it emitted a four-bullet revision
but missed the word budget and still omitted the explicit ticket-system
confirmation. Its reasoning repeatedly miscounted words and stopped while
editing. The recorded reward was zero; the old fact metric reported `0.75`,
which must not be read as semantic completeness. Trace SHA-256:
`f124fe7bbd6eb17b690077b459ffa370e4d25a7670a88e3bd26cb0b4151e1088`.

These results show that the original SFT representation did not transfer the
revision behavior reliably. That dataset placed the source, draft, and
feedback together in a new user message, unlike the live assistant-draft then
user-feedback prefix.

## Interpretation

The checkpoint can read these development chapters and often produce a strong
initial draft. The historical operations reward overstated completeness, and
the first one-step update did not reliably teach revision. The next bounded
intervention therefore uses the real role sequence: source request, prior
assistant draft, measured user feedback, then corrected assistant response.
The prior assistant draft is explicitly zero-loss context; only the correction
is supervised. With thinking enabled, Qwen3.5's generation prompt and complete
teacher-forced render diverge at the final newline BPE boundary; the existing
trajectory convention masks their exact common prefix and supervises the
complete-render suffix. This approximation and the exact token counts are
audited before training, and a live transfer check remains mandatory after any
update.
