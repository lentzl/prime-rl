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
