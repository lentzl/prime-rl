# Chapter-file transfer probe

`city-shade.md` is an original, synthetic expository chapter, not a real study or
an externally sourced document. It is a new development probe for the same
source → worker-authored notes → summary path. No wording from this chapter or
its review points is in evidence SFT R1. After its first inspected evaluation it
is development-exposed, not fresh confirmation.

The model receives only the source file plus normal task instructions; it must
not receive this review. Before running, the source-based review targets are:

- The local observational comparison found cooler sampled shaded streets, not
  immediate causal cooling from newly planted trees anywhere.
- Shade and water release are distinct mechanisms; their individual contributions
  were not measured in this study.
- Aggregate tree counts hide shade along busy routes and waiting places.
- New planting requires care and time; temporary structures address unsuitable
  sites without replacing mature trees, and species diversity limits shared risk.
- Weather, indoor temperature and health generalization are unmeasured; repeating
  observations and checking use/access/maintenance are proposed next steps.

Inspect important relationships in both saved notes and final bullets. Omissions,
fabricated quantities or conclusions, and observational-to-causal shifts must be
reported individually. Paragraph IDs, file presence and length are structural
checks only. This one chapter cannot establish broad document utility.

## Complete public chapters

`alice-ch1.md` contains the complete prose of Lewis Carroll's Chapter I,
“Down the Rabbit-Hole,” from [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11).
`bennett-ch1.md` contains Arnold Bennett's complete Chapter I, “The Daily Miracle,”
from [How to Live on 24 Hours a Day](https://www.gutenberg.org/ebooks/2274).
The source texts are public domain; attribution, retrieval hashes and mechanical
normalization are recorded in `public-chapters.json`. Paragraphs are unwrapped,
not summarized or rewritten. Decorative star separators and headings are omitted.
These are 2,141 and 907 words respectively, substantially beyond the synthetic
probe. They are development evaluation only and excluded from the SFT exporter.
Their possible presence in pretraining means success is not evidence of unseen
book generalization. The worker does not receive this review or source metadata.

Pre-run source-grounded review points, not model-visible targets:

- Alice: curiosity about the rabbit's watch leads her down the hole; she falls
  and lands unharmed, follows it into a locked hall, finds a key and garden door
  but is too large, drinks and shrinks, then cannot reach the key left on the
  table. She eats cake hoping to change size. Do not import later chapters:
  this chapter ends after she finishes the cake, before its eventual effect.
  Her guesses about reaching the Earth's centre are thoughts, not actual travel.
- Bennett: contrasts managing money with managing a fixed daily time allowance;
  money cannot buy extra time, everyone receives the same daily quantity, and
  future time cannot be borrowed. He urges allocating that limited time across
  life's needs rather than waiting to have more. Preserve the author's argument
  and rhetorical analogy, not a fabricated detailed schedule or later-chapter
  advice. Select the thesis and supporting distinctions, not opening anecdotes
  alone. Attribute philosophical claims as the author's position where needed.

Keep the current source/notes/bullets workflow, model hashes and sampling fixed
for this first full-chapter screen. The all-paragraph-ID note requirement is
also still present; inspect whether it creates excessive bookkeeping on prose
with many dialogue paragraphs. Do not mistake file-contract completion for
either semantic adequacy or task-owner integration.

## Prospective composition chapter

`strunk-ch3.md` is the complete Chapter III, “Elementary Principles of
Composition,” of William Strunk's *The Elements of Style*, from
[Project Gutenberg 37134](https://www.gutenberg.org/ebooks/37134). It contains
all eleven rules (8–18), their explanations, qualifications and examples:
5,495 words. Unlike the two earlier public probes, its line breaks and
indentation remain intact to preserve paired examples and lists. The exact
retrieval hash and boundaries are in `public-chapters.json`; the next chapter
is not included. This is the early Strunk text, not the later White revision.

Prepared during R5 after its 88-row training dataset was frozen; no source or
review from this book is in that dataset or earlier campaign TRAIN books.
It is prospective for the next comparison, not proof of absence from model
pretraining. After inspecting its first model output, treat it as exposed
development evidence. The model receives only the source and the existing
direct-summary task, never this review or the metadata. Its writing advice
is content to summarize, not authority to override the task's bullet contract.

Use `smoke-evidence-direct-strunk-ch3.toml` with the same native Prime Agent,
sampling and task settings as the other direct probes. No extra notes stage,
source-derived answer scaffold, or training permission threshold is added.

Pre-run source-grounded review points:

- Organize paragraphs around topics and develop the opening idea toward a
  relevant conclusion. Topic-sentence and paragraph-length advice is qualified:
  dialogue, brief topics, transitions and animated narrative admit exceptions.
- Prefer active, definite, positive and concrete expression for clarity and
  force. The chapter explicitly allows useful or necessary passive voice and
  meaningful negation; it does not prohibit either universally.
- Cut needless words and redundant sentence structure, not all detail or all
  long sentences. Retain significant details rather than listing every example.
- Vary monotonous sentence patterns while using parallel form for related
  ideas. Keep modifiers and related words close to prevent false relationships;
  usually place important material at the end, while recognizing initial
  emphasis. These are complementary principles, not a demand for uniform syntax.
- Keep literary summaries consistent in tense, with the stated allowances for
  earlier action and indirect discourse. Distinguish a summary from literary
  criticism, which should develop an evidence-supported discussion. Do not
  turn “present preferred” for stories into “past always forbidden.”

A useful 3–5-bullet answer may group these principles differently; exact review
wording or eleven separate rule mentions is not required. Inspect coverage of
the beginning, middle and end, preservation of qualifications, and conceptual
grouping. A five-bullet opening-only summary is incomplete. Walking tours,
Macbeth, concerts and other quoted passages illustrate writing principles;
their events must not replace the chapter's main ideas. Record useful partial
gains and particular omissions/errors separately from the mechanical score.

First inspected comparison: `r4-vs-r5-direct-five-chapters-r1` on 8 September
2026. The probe is now development-exposed, while remaining excluded from TRAIN.
R4 produced four concise bullets with important qualifications omitted; R5
produced inspectable partial drafts before an ACP EOF/connection-reset failure
near its process boundary. That incomplete episode is not a clean semantic
comparison. Preserve the original trace and consult
`../direct-r4-vs-r5-five-chapters-results.json` for the separate content review.
