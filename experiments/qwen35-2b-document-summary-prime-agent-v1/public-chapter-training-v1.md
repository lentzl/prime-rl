# Real-chapter direct-summary supervision

Prepared 2026-09-08. This is source material and assistant-authored supervision, not a trained checkpoint, independent human gold, or model-generated replay.

R3 builds on R2 and uses the existing native Prime Agent/IPython route to read a complete chapter and write 3–5 English key bullets. No intermediate notes file or paragraph-coverage obligation is intended. The earlier direct evaluation with wider correction allowance still produced factual errors and omissions; see `direct-public-chapters-r2-results.json`.

## Initial sources and review

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

Completed: native direct-summary export with observed runtime context, reproduced file observations, assistant-only supervision and full untruncated chapter tokenization audits, and registration in the existing training wrapper. R3 completed 32 full-dense updates from R2 on 20 retained TRAIN cases interleaved with these 20 public chapters. See `direct-sft-r3-run.json` for the checkpoint and receipts. The completed R2/R3 comparison is reviewed in `direct-r2-vs-r3-four-chapters-results.json`; no semantic promotion follows from training or formatting checks alone.

Owner direction, 2026-09-08: favor more successive weight updates, including learning from small gains and corrected failures, and accept behavioral-collapse risk. R3 became the experimental training frontier at that boundary, not a promoted reliable model; subsequent R4/R5 development continues this lineage. A poor held-out result must not impose a capability threshold before further training. Continue building on the latest valid checkpoint, preserve recoverable predecessors, and use comparisons to steer training and detect regression rather than veto all updates until the whole task passes. No automatic rollback follows from behavioral regression alone.

There is no campaign GPU-hour ceiling. Use the existing two-GPU host efficiently; additional infrastructure remains outside the current authorization. The archived diagnostic reference is recoverable locally; current learned checkpoints remain on the host.

## Further chapter teaching prepared during R5

R4 added Time Machine 5, Treasure Island 5, Flatland 17 and How We Think 9.
R5 trains from R4 on the resulting 44 direct episodes plus 44 authored repair
episodes with the poor draft and premature stop masked from target loss. The
completed R3/R4 semantic comparison is in `direct-r3-vs-r4-four-chapters-results.json`.

The next prepared teaching block adds four complete, source-reviewed chapters:
Time Machine 6 (2,183 words), Treasure Island 6 (1,949), Flatland 18 (1,685) and
How We Think 10 (2,690). These add 8,507 source words, for 28 public chapters
and 74,455 source words. `public-chapter-teacher-r3-additions.json` preserves
the previous four additional labels unchanged and adds these four new labels.

The new teaching signal is not merely more copies: the Traveller explicitly
qualifies his apparently complete causal theory as wrong; Jim's packet is now
actually opened while the expedition remains a plan; the Square's new visual
access does not reveal the Sphere's interior or establish moral superiority;
and Dewey argues that theoretical thinking complements rather than replaces
practical capacity. All chapter prose was read before these targets were
authored and checked. No later-chapter events are positive targets.

Prepared sources: `outputs/summary-public-training-chapters-r3/`.
Export: `outputs/summary-direct-sft-r4-chapters/`, with 96 episodes: 20 retained
TRAIN cases, 28 public chapters and 48 authored format-repair sequences. Every
one of R5's 88 complete parquet rows and case records is unchanged when matched
by identity; the eight new rows represent four new chapters in two states.
All 24 earlier source files and their chapter metadata are also unchanged.

The preparer now excludes trailing part-transition headings for either colon
or double-hyphen notation. Its initial Treasure Island VI extraction included
the subsequent part heading; this was corrected before dataset export, while
preserving the complete chapter ending. No training used the initial extraction.

The export passes the existing dataset validator, hashes, bullet constraints
and explicit message-mask checks. The real R5-tokenizer/trainer audit also
passed on the idle host: all 96 rows, 688,002 tokens, maximum 13,891, 28,834
supervised tokens, no truncation or source/user/incorrect-prefix loss. R6 is
training from R5 for another 128 updates; its first updates are verified, not
a claim of completed training or model improvement. See
`direct-sft-r6-run.json`. Alice, Bennett, Strunk and synthetic evaluation
chapters remain excluded from TRAIN; the Strunk probe adds no teacher targets.

The [standing charter](../../docs/continual-domain-training-charter.md) governs
subsequent curation, updates and recovery; preserving this block does not
require future datasets to be append-only.

## Next source block prepared during native-owner R1

The four new complete chapters are Time Machine 7 (2,321 words), Treasure
Island 7 (1,816), Flatland 19 (2,403), and How We Think 11 (3,360). Together
they add 9,900 source words: 32 public chapters and 84,355 public source words.
Each complete chapter was read before its five-bullet target was authored and
checked against the source. The review is by the authoring assistant, not
independent human gold or observed student success.

The targets preserve suspected versus confirmed locations, preparations versus
completed voyages, demonstrated versus conjectured dimensions, and practical
empirical success versus causal explanation. They summarize the main argument
or sequence without importing later events or turning historical assertions
into modern guidance. In particular, the four-dimensional figure's sixteen
vertices and eight bounding cubes are the Square's proposed analogy, not an
observed journey. Dewey's contrast does not dismiss all experience as useless.

`public-chapter-teacher-r4-additions.json` retains the earlier eight additional
labels exactly and adds four new ones. `outputs/summary-public-training-chapters-r4/`
contains the prepared sources. All 28 earlier source files and chapter metadata
are unchanged; both ends of the four new chapters were inspected. Sources remain
the same four TRAIN books; evaluation books and chapter outputs are not added.

`outputs/summary-direct-sft-r8-chapters/` contains 118 episodes: 20 retained TRAIN
cases, 32 public chapters, 52 authored format repairs and 14 semantic repairs.
All 110 R7 parquet rows and case records are unchanged when matched by identity.
The eight new rows teach four new chapter summaries in direct and correction
states; the existing exporter masks the bad draft and premature stop. Keeping
this rehearsal for this block is a choice, not an append-only corpus requirement.

Existing dataset validation, source hashes, summary constraints and mask
annotations pass. The existing file-observation replay function reproduces all
118 teacher file operations and Python result observations. These are local
preparation checks, not the real trainer/tokenizer audit or live native results.
The real R7-tokenizer/trainer audit subsequently passed on the idle host after
owner R1 completion: all 118 sequences, 839,107 tokens, maximum 13,891, 35,817
supervised tokens and 21,524 incorrect-prefix context tokens masked. No source,
user, tool or incorrect-prefix loss; no truncation or CUDA initialization.
Audit SHA256: `4d0a45a0b010e71ce3df530e4ea81e6c532e217c859dedfe4709625d24597452`.
The data and audit are copied to the host's separate
`/home/ubuntu/rlm/data/q35-2b-document-summary-direct-sft-v1-r8-chapters` directory
with matching hashes. R8 subsequently completed 128 updates with a verified
checkpoint; see `direct-sft-r8-run.json`. The native owner/R7 trial is reviewed
in `owner-r1-r7-direct-public-r1-results.json`.

Prepared manifest SHA256:
`55f149075867300b94cbc8244912d96520385975e0e3494a6e28cf5413227b63`.
Parquet SHA256:
`227b92714d1d0cf3dcc6aad30f3d571d8a3d4f0b278088f88a21056286318fea`.
Cases SHA256:
`303fe60afbce2219b09365b03e633ce146c59de1f403cab2f4b668d8fed16d25`.

## Chapter and semantic refresh prepared during owner R2

Four further complete TRAIN chapters add 10,714 source words: Time Machine 8,
Treasure Island 8, Flatland 20 and How We Think 12. All four chapters were read
and their boundaries checked before authoring and reviewing the summaries.
The 36 public chapters now total 95,069 words. This expands material and
relationships within the same four books, not the number of books or genres.

The lessons preserve provisional theories, actual versus suspected actions,
character confidence versus verified innocence, dreams versus waking events,
and the qualifications of an educational argument. Each new chapter also has
an explicitly authored wrong-but-format-valid draft and a source-checked
correction; wrong drafts and premature stops remain masked from target loss.
These are teaching examples, not sampled student trajectories or human gold.

`public-chapter-teacher-r5-additions.json` preserves the earlier 12 added
targets and adds four. `semantic-repairs-r2.json` preserves the earlier 14
repairs and adds four. All 32 previous source files and their metadata are
unchanged. Source files are in `outputs/summary-public-training-chapters-r5/`.

The refreshed `outputs/summary-direct-sft-r10-chapters/` contains 208 episodes,
including every one of the previous 191 rows and case records unchanged.
The 17 additions cover direct writing, format and semantic correction, native
child completion and one further character-count correction. All file episodes
replay and mask annotations pass. The real complete-sequence/token-loss audit
awaits the next idle boundary; no new weight update or capability gain is claimed.

The next worker R9 update can use this refreshed mixture from completed R8,
after that audit, then return to the combined native system. The artifact's
`r10` name is a dataset version, not a requirement for an additional training
run. This prospective curriculum choice does not change the live owner R2 run.
Detailed preparation and hashes: `native-chapter-refresh-preparation.json`.
