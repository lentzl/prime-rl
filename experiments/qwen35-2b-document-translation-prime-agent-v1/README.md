# Prime Agent document translation v1

This is one application smoke, not a new training campaign. It tests whether the
existing coordinator and terminal-worker lineages can produce a complete, useful
English-to-German document artifact through the native Prime Agent harness.

The root Prime Agent session is the document owner. It sees a structural index and
glossary, delegates three coherent source jobs, waits for explicit child reports,
then validates and persists the assembled artifact. Each child remains a Prime Agent
session and reads only its assigned job. The authored German reference is host-side
diagnostic data and is absent from all runtime files and prompts.

`tools/docflow_v1.py` is intentionally not part of this runtime. Stable IDs, source
hashes, coverage, and artifact validation are implemented directly in the small
Verifiers task contract. We will add extraction or assembly helpers only when a real
document exposes that need.

Starting models:

- document owner: the strongest existing e33-descended coordinator checkpoint;
- terminal translators: protected H176 worker lineage;
- no weight updates in the first smoke.

The next training decision follows the observed failure: translation quality points
to the worker; orchestration, fan-in, or artifact-completion failure points to the
owner or harness contract. We do not train both roles preemptively.
