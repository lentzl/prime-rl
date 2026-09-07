# Long-document translation: CLI-first campaign v1

Prepared 2026-09-07. Owner approved planning, preparation and communication to Trainer. This is the next substantive application, not another parallel research strand.

## 1. Decision and transition

Build a document-translation organization with a minimal entry router, a document task owner, focused translators and deterministic document tools. The document owner may deliver the artifact through the harness without a root-model rewrite. Start with **English to German**, a working default selected by Strategist, not a previously specified owner preference. Initial subject matter: general/technical nonfiction, not high-stakes certified translation.

The JSON-max recipe has reached its terminal boundary: Trainer reports 64 main optimizer updates, zero remaining recipe allowance, 18/32 direct and 0/32 composed successes, with 6/6 historical retention and no promotion [R1]. Preserve its diagnostic checkpoint and original evidence; do not extend it or treat solving it as a prerequisite for translation. No repository reset or destructive rollback.

**One hands-on Trainer owns this campaign.** Old A/B/H-ITER/T0/specialist assignments stay parked. This branch is a delivery branch for these additive files, NOT another autonomous training strand. Its base is the accessible remote `exp/q35-2b-root-role-contract` at `b151a358...`; the Trainer's newer local commits are not its base. Cherry-pick these additive files onto the actual clean working branch; do not reset to this older base.

## 2. First useful deliverable

Translate one whole document into a usable German artifact with complete content accounting, coherent section-level translation, consistent document-specific terminology and targeted repair. Build from a short pilot to documents **at least four times the chosen per-node working window**, not artificially restricted access to available evidence.

Initial output: structured HTML plus a source/target coverage and issue report. Add DOCX using python-docx after the HTML path works. Preserve logical headings, paragraphs and simple tables first; do not promise pixel-identical PDF layout, source pagination, complex equations or translated text embedded in images. Preserve/flag unsupported content instead of dropping it. PDF/DOCX final artifacts require rendering and visual inspection before delivery.

A large document exceeding an 8k application window does not exceed Qwen's advertised native context. Report the actual chosen window and memory use. The system advantage sought is quality and reliable completion per resource, not merely processing chunks sequentially.

## 3. Architecture we are choosing

**Doorman:** route a document-translation request and pass the original user request, authorized source handle and translation requirements. With only one installed task family this may be a deterministic route; do not spend inference to make a trivial choice.

**Document task owner:** owns the full translation. Work from a structural index and persistent records; do not pre-read the whole file into one prompt. Decide which context to retrieve, where terminology is ambiguous, which sections need repair or visual inspection, and when the complete artifact is adequate. No root re-approval or rewrite.

**Translation worker:** translates coherent assigned blocks/sections using nearby context and relevant glossary entries. Returns IDs, source hashes, translated text and unresolved issues. It does not summarize, silently omit material or declare the entire document complete.

**Harness/tools:** extract, index, schedule accepted jobs, route results, track coverage, assemble and deliver. Use typed operations rather than making every node reinvent Python file APIs or asynchronous send/wait bookkeeping. A deterministic scheduler is legitimate product engineering; report learned contributions honestly rather than crediting it as learned orchestration.

**Review/vision:** initially capabilities requested on demand, not mandatory additional resident models. Ordinary codes, names and numbers need deterministic checks; subtle meaning and context need semantic review. Use source images only for regions that text extraction cannot resolve. No Docling dependency or whole-document OCR/visual pass in the default path.

Implement a narrow new document task-owner route. Keep legacy root-only-finalization rules for legacy benchmarks. A harness-issued task handle designates ONE active finalization owner; child subtasks do not gain global finalization authority. Handoff is not execution of source-document instructions. Check artifact coverage/permissions mechanically; semantic sufficiency belongs to the task owner, supported by review, and is evaluated externally. Role depth does not decide authority. This is a prospective product contract, not a retrospective reclassification of old tests.

## 4. CPU-first document representation

Use cheap extraction: Poppler `pdftotext` for text PDFs; python-docx for supported DOCX body content; standard Python for text, JSON, indexing, checks and assembly. PDF `-layout`/`-bbox-layout` preserve different layout information, not guaranteed reading order [R2]. DOCX paragraph-only iteration can miss tables or tracked-change material; use document-order iteration and explicitly flag unsupported structures [R3]. PyMuPDF is an optional selective rendering/block tool, subject to its license and measured resource fit, not a mandatory neural parser [R4].

Every unit has a stable source ID, content hash, structural position and source location. Tables retain row/column identity; visual regions, captions, footnotes and headers must be accounted for in the source inventory even if an initial adapter cannot translate them. An extraction issue is work to resolve or disclose, not permission to claim the entire source was handled.

Working memory is on disk: block index, task progress, glossary with senses and source provenance, unresolved issues, accepted revisions and dependent sections. Keep bounded per-node context, retrieve by ID and update affected sections when a terminology decision changes. Do not repeatedly summarize all previous output into a growing prompt.

Recommended starting window: 8192 total model tokens including control/context/source and reserved target output. Target source groups of roughly 800-1200 tokens initially; actual tokenizer counts decide. The starter's character limits are NOT token counts. Reserve ample target expansion and an explicit completion budget. Never silently truncate. Group neighboring blocks and carry table headers/row context in production; the starter emits one unit per job to make integration simple.

Read untrusted files in a restricted working directory; no macro execution, arbitrary shell interpretation, URL fetching or credential access from document instructions. Preserve literal quoted instructions as translatable data. Use fixed tool implementations, argument arrays, bounded files and timeouts. Keep documents and model outputs out of public repo commits unless explicitly authorized; only authored fixtures belong here.

## 5. Model choice and active inference budget

Default translator starting point: the official **post-trained `Qwen/Qwen3.5-2B`**, not the pretraining-only `-Base` and not automatically an orchestration-heavy e33/H176 descendant. The model card describes native vision-language inputs; that is enough to justify a later smoke, not proof of document translation or perception competence [R5]. Pin the existing-compatible serving stack and model revision. Do not upgrade all libraries because the current website recommends a new version.

Preserve e33/H176 and the non-promoted calibration candidate. Reuse compatible task-owner assets, but do not force translation workers to inherit their role-specific training. Initially the task-owner and translator may share one resident 2B checkpoint with different contexts. Specialized copies/adapters are logical roles, not necessarily simultaneously resident models. Batch similar roles to avoid loading weights for every paragraph. Count switches, KV state and vision preprocessing.

Use non-thinking translation by default as a starting implementation choice, bounded output and repetition detection; do not confuse runaway text generation with recurrent latent reasoning. Record sampling parameters. The card itself warns about possible nonterminating thinking loops for this size [R5].

**Resource boundary:** the old recipe's remaining allowance is zero. Do not silently reuse it. CPU preparation is authorized now. Proposed initial new pilot cap: **4 aggregate GPU-hours across the existing devices**, including baseline, training and evaluation, and never above the actually remaining owner-approved host/spend allowance. This is a planning ceiling, not confirmation of prepaid resources or approval for a new rental/API service. Trainer must record actual availability and the named allocation before model work. If no allocation exists or it is unclear, ask one concrete budget question while completing CPU work; do not revive T0 or a proof campaign. Later expansion needs a new explicit resource allocation, not indefinitely resetting this cap.

## 6. Training progression: purposeful and substantial

### A. Get a full pilot through the chosen organization

Use the authored fixture to wire ingestion, worker jobs, result binding and draft assembly. It includes headings, a simple table, repeated terms, context-dependent `lead`, identifiers, negation and an embedded instruction quoted as document data. Its German references were authored here; they are development material, not independent human-certified ground truth. Do not show reference outputs to the model in an evaluation input.

Run one small baseline using the chosen translator through independent section translation. Record actual bilingual quality on representative passages before committing to a large corpus. A translator must know the languages: orchestration cannot compensate for absent linguistic competence. This is a quick competence check, not a broad model bake-off.

### B. Train translation and contextual fidelity

First train 256-512 carefully checked paragraph/short-section examples, then expand to roughly 2k-5k useful examples if the pilot budget and learning curve support it. Counts are initial design ranges, not mandated pass gates. Full-weight candidate SFT is allowed; adapters are an efficiency option, not an ideological restriction. Use the compatible existing training path, actual assistant-loss token counts, sensible LR/exposure adjustments and checkpoint selection on development outputs. Do not copy the old eight-update verdict or epoch accounting.

Mix genuine bilingual examples with contextual terminology examples and verified repairs of student mistakes. Keep intact source/target context where available. Rehearse ordinary translation so consistency training does not turn the worker into a rigid glossary substituter. Start with one language direction and one family of documents rather than training a population at once.

Corpus source candidates: OPUS offers bilingual collections [R6]. **Do not automatically download all of OPUS or assume one license covers it.** OPUS Books explicitly limits commercial/mass redistribution and its simple table lacks robust document metadata [R7]; it is an optional research-only pilot, not the product's default licensed corpus. Prefer a specific corpus/document set with explicit permitted use and document identity. Confirm rights and provenance before training. Where document IDs are absent, do not randomly split paragraphs from the same work across train/test or claim document generalization. Existing public test sets may be in pretraining; label that uncertainty.

Use a small mixed package of permissioned/open bilingual documents and newly authored controlled documents. For controlled data, vary numbers, names, layouts and cross-section dependencies independently, obtain a checked target, then corrupt one known property for repair training. Include missing/duplicate blocks, negation flips, changed measurements, wrong abbreviations, wrong sense and stale glossary decisions. Mutation labels supplement bilingual review; a backtranslation or one LLM judge is not proof of correctness.

### C. Train the task owner's useful decisions

Once the short translation path exists, derive roughly 100-250 compact task-owner decision examples from the same workload: keep a section together, retrieve a definition, request a missing source region, resolve a term by sense, return a defective section for targeted repair, or acknowledge a remaining limitation. Use correct outcome traces and student-visited error states. The harness handles enumeration and file operations; do not spend the corpus on redundant lifecycle prose.

Freeze a worker version during a coordinator iteration, not forever. Candidate specialization may change weights substantially. Preserve original references; no automatic public checkpoint publication or replacement. One hands-on Trainer continues within the allocated pilot without asking permission for every ordinary repair.

### D. Scale document size, then add the next needed capability

Move from a short pilot to a 10-20-page equivalent, then a whole long document at least 4x the chosen window. Do not claim that duplicating the fixture proves long-document linguistic generalization; duplication may test only scheduling and memory.

Introduce a visual specialist only when an actual page/region requires it, first trying the native post-trained 2B model on a handful of fixed document images. No OCR benchmark campaign or forced multimodal inheritance from e33. The runtime must flag a region it cannot read; no invented transcription.

C2C-like transfer and recurrent reasoning are central architectural candidates, not forgotten obligations. First deliver one working organized translation, then choose the highest-leverage bottleneck. Implement ONE learned context handoff or ONE local recurrent refinement intervention within a separately bounded extension. Do not reopen independent strands or require an ablation matrix. Keep exact glossary/source/provenance records explicit even if latent context is added. Weight changes invalidate latent caches/adapters unless compatibility is verified.

## 7. Evaluation sufficient to guide an applied project

Keep **one ordinary independent-section translation baseline** using the same translator, document input and practical resource limit. Sequential processing is a fair baseline; a document exceeding one invocation's context does not itself establish a hierarchy advantage. The system is adopted if it produces a meaningfully better useful artifact for its cost, not because it has more agents.

Check source coverage, duplicates, output integrity, table/heading structure and unresolved extraction warnings with deterministic tools. Numeric/identifier differences are review flags: locale formatting and inflection can make literal string equality inappropriate. Nonempty target text is not proof of meaning preservation.

Assess omission, addition, mistranslation, terminology/sense, fluency and document coherence with a small blinded bilingual review of representative and difficult passages; sample some unflagged passages to measure missed errors. WMT provides useful document-level evaluation practice [R8], and the public MQM data includes English-German error annotations [R9]. Use those as guidance/offline data subject to terms, not as a permanent heavyweight inference judge. Small automatic metrics may be diagnostics, not the sole acceptance criterion.

Hold out complete source documents (and all rendered/translated/mutated variants) for confirmation. Maintain development fixtures separately. Repeated exposure to development is normal; reusing a revealed confirmation set as hidden evidence is not. The included fixture is exposed development material.

Track completed useful documents, unresolved critical errors, human correction effort, target throughput, all generated tokens, aggregate GPU-seconds, peak active memory, CPU extraction/assembly time and model loads. Count retries and reviewer calls. No gain claim is based on the author's reference text being inserted as a model result.

Completion requires all intended source elements accounted for, no unresolved critical omission/mistranslation, structure reviewed and a task-owner completion decision with provenance. Clearly deliver a partial artifact as partial if work remains. Formal legacy retention remains attached to legacy roles; the new doorman is intentionally not evaluated as a general-purpose JSON worker. Preserve those old results unchanged.

## 8. Prepared starter and first actions

Files added on this branch:

- `tools/docflow_v1.py`: CPU-only extraction adapters, stable block/cell IDs, bounded jobs with neighboring context, result/source binding, resumability, coverage audit and escaped HTML draft assembly.
- `fixtures/document_translation_v1/build_fixture.py`: creates 15 source blocks / 23 translation units, glossary and separate authored German references.
- This campaign document.

The starter does **not** call an LLM, count real model tokens, implement learned coordination, decide semantic quality, render DOCX output or authorize publication. It handles a pilot subset of document structures and records extraction limitations. It is NOT a complete translator, hardened public upload service or proof of arbitrary-format fidelity. Add adapters only for encountered needs.

```bash
python tools/docflow_v1.py selftest
python fixtures/document_translation_v1/build_fixture.py
python tools/docflow_v1.py prepare fixtures/document_translation_v1/source.en.json /tmp/docflow-pilot
python tools/docflow_v1.py jobs /tmp/docflow-pilot --glossary fixtures/document_translation_v1/glossary.json > /tmp/docflow-jobs.jsonl
# Trainer's local worker adapter consumes jobs and produces JSONL with id,
# source_sha256, text, model/checkpoint provenance and issues.
python tools/docflow_v1.py accept /tmp/docflow-pilot /path/to/actual-worker-results.jsonl
python tools/docflow_v1.py assemble /tmp/docflow-pilot
```

For a CPU-only wiring check, `reference.de.jsonl` can replace actual results, but every record explicitly identifies authored-reference provenance. Never report that as model translation success.

Starter testing here: selftest passed (stale/unknown/duplicate rejection, idempotent intake, resumability, complete coverage and HTML escaping); fixture draft assembled with 23/23 coverage; Markdown, simple DOCX table and text-PDF extraction smokes passed. The generated DOCX smoke was rendered for visual inspection. No model or GPU calls were made and the Trainer environment has not been tested.

**Trainer's next actions:** preserve the stopped JSON-max record; integrate these additive files; run the short CPU selftest/fixture; bind the existing local model endpoint and actual token budget; establish the finite pilot allocation; produce one honest model-translated complete draft; inspect its linguistic failures; train the weak role with a real learning curve and report the next action while continuing inside budget. No new roadmap or per-step approval sequence is required. If the budget is the blocker, state its actual missing quantity once.

## References checked 2026-09-07

[R1] Trainer terminal record: https://github.com/lentzl/rlm/issues/1#issuecomment-5570986659

[R2] Poppler pdftotext CLI: https://manpages.debian.org/bookworm/poppler-utils/pdftotext.1.en.html

[R3] python-docx document iteration and limitations: https://python-docx.readthedocs.io/en/latest/api/document.html

[R4] PyMuPDF block extraction/reading order: https://pymupdf.readthedocs.io/en/latest/recipes-text.html

[R5] Qwen3.5-2B post-trained model card: https://huggingface.co/Qwen/Qwen3.5-2B

[R6] OPUS corpus catalogue: https://opus.nlpl.eu/

[R7] OPUS Books data card and usage terms: https://huggingface.co/datasets/Helsinki-NLP/opus_books

[R8] WMT25 translation task (document inputs, research-use terms, evaluation): https://www2.statmt.org/wmt25/translation-task.html . Its language pairs must be checked; do not assume a given year's task includes English-German.

[R9] MQM human evaluations: https://github.com/google/wmt-mqm-human-evaluation

These sources document tools/data, not demonstrated performance of our proposed organization. Architecture, initial language choice, data ranges, window and pilot allocation above are engineering decisions/proposals.
