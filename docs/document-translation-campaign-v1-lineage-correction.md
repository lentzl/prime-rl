# Document translation campaign v1 — starting-lineage correction

2026-09-07. This note **supersedes the model-starting-point language** in `document-translation-campaign-v1.md` wherever the two conflict.

## Correction

The document application is an **extension of the delegation system we have already trained**, not a restart from a plain post-trained Qwen3.5-2B organization.

Default starting lineages:

- **Root/doorman:** candidate copy from protected coordinator **e33**. Narrow its learned responsibility toward initial handoff/routing; do not retrain a vanilla model from scratch to rediscover delegation.
- **Document task owner / recursive document coordinators:** candidate copies from **e33** unless a later promoted descendant clearly dominates it. Preserve and extend the recursive coordinator behavior already established.
- **Terminal translation worker:** candidate copy from protected worker **H176**. Train translation competence into that worker lineage while rehearsing the terminal worker/return contract. Do not discard its learned child-role behavior merely because the task changed.
- **Additional specialists:** preferentially branch from the closest already-capable lineage (coordinator or terminal worker) and specialize it. A clean post-trained Qwen3.5-2B checkpoint may be used as a diagnostic language/vision reference or, later, as the ancestor of a specialist only if evidence shows the existing lineage has materially lost the needed base capability. It is **not** the default system starting point.

Protected e33/H176 remain immutable references. Training occurs on descendants/copies.

## Architectural reason

The project has already paid substantial compute and engineering cost to teach small nodes the mechanics and semantics of recursive delegation, role identity, child return, and coordinator hierarchy. The document campaign should test whether **specialization plus organization can turn those learned primitives into useful system capability**. Starting the organization from vanilla Qwen would confound that goal with relearning the basic system contract and would waste prior work.

The new application deliberately changes role emphasis without discarding learned foundations:

```text
User
  -> e33-descended doorman
       -> e33-descended document task owner
            -> H176-descended translation worker(s)
            -> later specialist children as needed
       -> task-owner direct artifact delivery through harness
```

The doorman is intentionally narrower than historical e33: it should become excellent at initial delegation, not regain general local execution. The document task owner inherits coordinator/decomposition skills and is trained on document-specific organization, context retrieval, terminology decisions, repair, sufficiency, and downstream ownership transfer. H176 descendants inherit terminal worker discipline and are trained hard on translation/specialist competence.

## Training consequences

1. **First system smoke uses e33 + H176 lineages.** Wire the CLI document flow into those roles and obtain a complete model-produced short draft before changing ancestors.
2. **Train H176 descendant for EN->DE translation**, with terminal-role rehearsal. Translation data should change weights substantially if useful; H176 is a starting point, not a freeze constraint.
3. **Train e33 descendant for document task ownership**, using real document episodes, task handoff, section/worker delegation, context retrieval, targeted repair, and direct completion semantics. Preserve recursive coordinator skills with a small rehearsal mixture rather than making the model redo JSON arithmetic.
4. **Train the doorman as a narrowed e33 descendant** only once there is a meaningful destination to route to. With one active application, routing may initially be deterministic in the harness; do not spend model compute proving a trivial route.
5. **No mandatory vanilla comparison.** If a quick frozen vanilla translation sample is almost free and useful for diagnosing whether H176 lost language competence, it may be run. It is not a required ablation and it does not define the runtime architecture.
6. **Vision later:** first smoke the retained multimodal behavior of relevant descendants on a few document images. If it is inadequate, branch or train a dedicated visual specialist. Do not replace the whole organization with vanilla Qwen to recover vision.

## Interpretation

The hypothesis is now sharper:

> Can the delegation-capable e33/H176 organization, specialized for a real long-document task, become substantially more useful than its individually weak 2B components?

The application should build on accumulated capability, not reset it.
