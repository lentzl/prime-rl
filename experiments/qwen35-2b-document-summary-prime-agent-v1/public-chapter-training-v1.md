# Real-chapter direct-summary supervision

Prepared 2026-09-08. This is source material and assistant-authored supervision, not a trained checkpoint, independent human gold, or model-generated replay.

The next training step should build on R2 and use the existing native Prime Agent/IPython route to read a complete chapter and write 3–5 English key bullets. No intermediate notes file or paragraph-coverage obligation is intended. The direct evaluation with wider correction allowance still produced factual errors and omissions; see `direct-public-chapters-r2-results.json`. Increasing the allowance again is not the next experiment.

## Sources and review

| Book | Chapters | Project Gutenberg ebook |
| --- | --- | --- |
| The Time Machine | 1–4 | [35](https://www.gutenberg.org/ebooks/35) |
| Treasure Island | 1–4 | [120](https://www.gutenberg.org/ebooks/120) |
| Flatland | 13–16 | [97](https://www.gutenberg.org/ebooks/97) |
| How We Think | 1–8 | [37423](https://www.gutenberg.org/ebooks/37423) |

All 20 complete extracted chapters were read before their summaries were authored and checked against the source. The corpus contains 55,997 source words and 2,460 summary words; summaries range from 102 to 157 words. `public-chapter-teacher-v1.json` records the bullets and specific review considerations. The review is by the authoring assistant, not an independent adjudicator.

The sources include narrative chronology, hypothetical versus actual events, chapter boundaries, explanatory arguments and qualifications. Summaries should capture the main points, not one obligation per source paragraph. Historical assertions are not automatically endorsed as current factual knowledge.

The evaluation books Alice (ebook 11) and Bennett (2274) are excluded as training sources, as are their evaluation summaries. This is not a zero-overlap claim: Dewey chapter 8 contains an incidental Alice cake allusion, which is retained in the complete source but not included in its teacher summary. Public-domain books may already occur in model pretraining.

The four catalog entries identify the works as public domain in the USA. Original downloads, including their full Gutenberg license text, are retained locally. No BookSum summaries or OpenStax text are included.

## Reproduction and checks

`scripts/prepare_document_summary_public_chapters_v1.py` pins the four raw download hashes and expected full-book heading counts. It extracts complete selected chapters, unwraps paragraph whitespace, omits chapter/inter-part headings, and retains prose, sidenotes, footnotes and illustration placeholders. It refuses changed raw files or an existing output directory.

Raw inputs: `outputs/summary-public-training-sources/pg{35,120,97,37423}.txt`.
Prepared sources and per-chapter hashes: `outputs/summary-public-training-chapters-r1/SOURCES.json` and adjacent Markdown files.

Run the script with `uv run python`, `--raw-dir` pointing to those raw inputs and `--output-dir` pointing to a fresh directory. Primary download URLs and raw hashes are pinned in the script and recorded in the generated manifest.

Checks actually run:

- All 20 slugs present exactly once; no evaluation ebook used as a source.
- Each source hash and word count match its manifest.
- Each summary has 3–5 single-line bullets, 5–45 words per bullet, below 80% of source length.
- Regenerating from pinned inputs reproduces every prepared file byte-for-byte.
- Existing-output protection and code lint pass.

Still required before an optimizer launch: export native direct-summary episodes using observed runtime context, reproduce their file observations, audit assistant-only supervision and full untruncated chapter tokenization, and register the dataset in the existing training wrapper. Preserve R2 and compare a resulting candidate against it; no semantic promotion follows from formatting checks alone.

There is no campaign GPU-hour ceiling. Use the existing two-GPU host efficiently; additional infrastructure remains outside the current authorization. The archived diagnostic reference is recoverable locally; current learned checkpoints remain on the host.
