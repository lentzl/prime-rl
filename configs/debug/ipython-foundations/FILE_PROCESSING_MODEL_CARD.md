---
base_model: lentzl/rlm-prime-agent-qwen35-ipython-recovery-r2-20260807
library_name: transformers
pipeline_tag: text-generation
tags:
  - prime-agent
  - ipython
  - tool-use
  - grpo
---

# Qwen3.5 IPython File Processing R1

This is the selected merged checkpoint from the fifth IPython curriculum rung. It
starts from `lentzl/rlm-prime-agent-qwen35-ipython-recovery-r2-20260807` at revision
`cb6acc9b6187b34645c83b9e5b876c2ea226bb9c`, adds two epochs of replay-mixed typed
file-processing SFT, and merges the third step of a bounded GRPO refinement.

The curriculum covers structured download results, retained file paths, parser
selection for text, Markdown, CSV, JSON, PDF, DOCX, and unknown inputs, plus
traceback-guided recovery from malformed, unavailable, empty, encrypted, and
incorrectly decoded inputs. It is intended for evaluation in the matching Prime
Agent persistent-IPython harness, not as a general chat model.

On 18 held-out file-processing tasks, the selected policy improved process score
from 0.103 at the parent checkpoint to 0.492, grounded-answer rate from 0.167 to
0.389, and observed processing outcomes from 0.167 to 0.667. Mean extra errors fell
from 4.444 to 1.556. Completion and cross-turn state-continuity checks remained
perfect over four samples per family.

Known limitations: exact final-answer accuracy on the held-out silent-assignment
family was 0.25 despite 1.0 silent-assignment recovery, and repeated calls increased
from 0.056 after SFT to 0.278 after GRPO. These remain hard regression gates for the
next curriculum rung.

Training and evaluation definitions are available on the
`exp/file-processing-r3` branch of `lentzl/prime-rl`, with the Verifiers environment
at commit `88e158af6f5a77cddd6a8dcff91291677deec02c`.
