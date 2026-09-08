# Evidence curriculum R1

## 2026-09-08 06:02 UTC

Status: launched; both training ranks initialized. No learning result yet.

- Source: retained v8, SHA-256 `2a8575a01814af2cd4914f9984d0001d909f89c1414534e90220b476df64b28c`.
- Code: `7f9d3b177`; dataset: `q35-2b-document-summary-evidence-sft-v1-r2`.
- Thirty-two phase samples from sixteen source cases: twelve earlier training
  chapters with repaired targets, plus two paired contrast families (four cases).
  The pairs deliberately share vocabulary while changing obligations; these are
  not sixteen independent domain families.
- Sixteen extraction and sixteen realization samples. In realization, prior
  extraction assistant messages have zero loss. Teacher notes appear only in
  authored TRAIN episodes, never in live evaluation.
- Exact-host dataset SHA-256: `c99ac5fc2fa3ae6a7ea5a1805400328e52d3a8125603c9ac1dcde9815172a1d2`.
- Mask audit: 160,243 rendered tokens, 7,747 supervised tokens across the corpus;
  only current-phase assistant messages supervised, no truncation, no CUDA use
  in the audit. Thinking-enabled rendering matches the tool-enabled episode mode.
- Four full-weight BF16 AdamW updates, LR `1e-6`, two GPUs, batch size 8,
  checkpoint only at update 4. Numerical dtypes are unchanged.
- Focused tests: 28 passed locally and on the actual host before launch; Ruff
  and the native SFT dry-run passed.
- Run directory: `/home/ubuntu/rlm/outputs/q35-2b-document-summary-evidence-sft-v1/h176-summary-evidence-sft-step4-r1`.
- Visible launch surface: `q35-spade-open:Launcher`.

After training: inspect optimizer health and the stable checkpoint, then run the
same unassisted source-notes-summary episode. Preserve extraction versus realization
inspection, structural versus semantic verdicts and source hashes. Do not promote
from training loss or artifact-presence reward. Broader fresh-document evaluation
and task-owner integration remain unfinished.

## 2026-09-08 06:20 UTC

Training completed all four updates with no NaNs. Loss fell from 1.0698 to
0.8541567; peak allocation was 11.2 GiB per device. The stable candidate hash is
`b8cd9656a6b61b23386bd50a24aac47a5175d565bbbc9b8ed80ba91d9dfc7202`.

The first live postflight regressed: all four source records were present in
memory, but repeated `write_text` calls overwrote earlier records. The saved
notes retained only one paragraph, and the model subsequently wandered into
goal/delegation calls and exhausted 16 turns without writing a summary. This is
a behavioral failure, not a provider failure; no promotion. The exact artifact
and message excerpt is in `evidence-sft-r1-postflight.json`.

Verifiers `48331983` now rejects notes missing source IDs before capture and
returns a concrete single-write repair instruction. Stage feedback names the
next file operation without the generic gate-program framing. Neither change
supplies semantic answers. Forty-two focused environment tests pass. Next is a
no-update rerun of this same candidate, followed by a matched v8 comparison if
the working-state path completes. Ordinary chapter-file input is also available;
structural coverage must not be presented as semantic correctness.

Owner resource instruction and Strategist correction #5580223241 supersede the
artificial aggregate GPU-hour ceiling. Optimize useful progress per rented
machine-hour on the existing two-GPU host. GPU time is telemetry, not an approval
gate. No extra machines, paid services, storage purchases or rental extension are
authorized. Disk has 7.3 GiB free after this checkpoint; protect retained evidence
and checkpoint references when preparing further training.
