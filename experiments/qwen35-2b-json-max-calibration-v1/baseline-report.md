# JSON-max calibration result

The matched calibration found a coordinator bottleneck, not a worker-computation
bottleneck. No model was trained and no candidate checkpoint was created.

## Result

- Matched raw-direct e33: 3/32 exact task successes (9.375%).
- Two-shard e33 + H176: 0/32 exact task successes.
- H176 child reports observed: 14/32 episodes.
- Numerically exact H176 reports: 14/14 observed reports.
- Strict child-contract completion: 8/32 episodes.

The two arms used the same raw JSON shards and oracle answers per task. The direct arm
gave both shards to e33; the composed arm gave one shard to e33 and the other only to
H176. Evaluation used fresh `valid_gen` indices 0–31 with master seed 20260819.

## Decision

Skip H176 training. When e33 successfully invoked H176 and received a report, the
reported maximum was always correct in this sample. The dominant failures occurred
before a usable child report or when e33 computed/combined its coordinator-owned
shard. e33 also failed 29/32 direct controls, so improving the already-correct H176
numeric behavior would not address the limiting component.

The next useful experiment should target the smallest coordinator behavior:
reliable local JSON parsing, one correct child spawn/yield/receive cycle, and exact
`max(remote_max, local_max)` aggregation. It should retain this paired surface and
reuse untouched H176 as the worker.

## Boundaries

- The prepared training corpus remains available but was not consumed by a trainer.
- The sealed `ood_gen` split was generated but not opened by evaluation.
- H176 and e33 were rehashed after evaluation and remain byte-identical to their
  protected checkpoint identities.
- This is a 32-pair calibration result, not a broad capability claim.

Machine-readable identities, metrics, and artifact hashes are in
`baseline-evidence.json`. Trace-level paired analysis is stored with the result at
`CALIBRATION_ANALYSIS.json`.
