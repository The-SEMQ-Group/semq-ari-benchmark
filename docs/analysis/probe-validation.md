# Probe choice and validation

A reproducibility probe must distinguish subject changes from changes in its own computation.
The [canonical specification](../../spec/ari-canonical-v0.1.md) selects SEMQ QBIN for embedding reports.
That selection does not establish that only SEMQ can serve as a probe.
See the [retraction ledger](../retractions.md).

## Fixed artifacts and repeated training

The seed comparison measured Hamming 0.0000 for SEMQ and approximately 0.5 for independently seeded PQ, OPQ, and LSH instances.
This measures reproducibility across independently constructed probes.
It does not establish failure of a fixed, published codebook.
Distribute and identify the actual probe artifact when its construction can vary.

## Floating-point reassociation

A vector quantizer selects the nearest centroid using computed distances.
Algebraically equivalent distance formulas can produce different floating-point values near a tie.
The real-codebook experiment tested direct and expanded squared-distance formulas on SciFact embeddings.

See [probe-verifiability results](../../experiments/probe-verifiability/RESULTS.md) for the complete table.
The reconstruction property held in every tested configuration.
Row disagreement ranged from zero to approximately five percent, depending on the configuration.
A constructed near-duplicate codebook produced much larger disagreement, but did not represent the fitted codebooks.

## Basis for the canonical choice

SEMQ encodes through threshold comparisons with a fixed calibration scale.
Its calibration artifact is smaller than a vector-quantizer codebook.
The operator and implementation still require verification under the intended platforms and input conventions.
The identity `encode(reconstruct(c)) = c` alone does not prove cross-platform encoding invariance.

A frozen vector quantizer remains a viable alternative for a separately defined protocol.
Its artifact, distance computation, and tie behavior need explicit verification.
The approximately 14-fold LSH sensitivity comparison is one synthetic result and requires replication.

## Detection scope

The [sensitivity sweep](../../experiments/drift-sensitivity/RESULTS.md) measures synthetic perturbations.
The [regime experiment](../../experiments/regime-discrimination/RESULTS.md) compares serving changes with retrieval measurements.
Code changes can coexist with unchanged Recall@10. Retrieval lists can still change.
Neither experiment establishes that only SEMQ can detect representation changes.
