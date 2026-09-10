# Findings

These results apply to the recorded inputs, model revisions, and environments.
Detailed methods, tables, and limitations belong to the linked experiments.

| Experiment | Finding | Evidence |
| --- | --- | --- |
| Embedding panel | The panel contains five hosted APIs and nine local models. Hosted ARI ranges from 0.169 to 1.000. | [Panel results](../experiments/deployed-agent-panel/RESULTS.md). |
| GPU determinism | TF32 changed code agreement in the tested configurations. Disabling TF32 restored the measured controls. | [GPU results](../experiments/gpu-determinism/RESULTS.md). |
| Retrieval comparison | Some serving changes altered codes while Recall@10 remained unchanged. Ranked retrieval lists could still change. | [Regime results](../experiments/regime-discrimination/RESULTS.md). |
| Internal decoding | Logit codes could change while generated text remained identical. | [Decoding results](../experiments/decoding-reproducibility/RESULTS.md) and [larger models](../experiments/decoding-reproducibility/RESULTS-production-models.md). |
| Detector baselines | The top-2 margin used less reference storage for the tested perturbations. It missed perturbations confined below the selected ranks. | [Baseline comparison](../experiments/decoding-reproducibility/BASELINES.md). |
| Drift profile | The rank profile tests whether measured changes affect low-ranked logits. | [Rank-profile results](../experiments/drift-rank-profile/RESULTS.md). |
| Hosted decoding | The dry run measured repeated-call variability and calibrated the concurrency procedure. | [Dry-run results](../experiments/arid-dry-run/RESULTS.md). |
| Harness effect | Cross-harness disagreement exceeded within-harness disagreement in the public trace dataset. | [Harness results](../experiments/harness-effect/RESULTS.md). |
| Probe verification | Frozen vector quantizers can satisfy the tested reconstruction property. SEMQ is not necessary for the index. | [Verification results](../experiments/probe-verifiability/RESULTS.md). |

## Interpretation limits

- Code equality measures representation agreement, not answer quality or semantic equivalence.
- A hosted API condition can expose several internal changes. It does not identify their individual causes.
- Identical Recall@10 values do not establish identical retrieval lists.
- Synthetic sensitivity does not establish the frequency of real deployment changes.
- Historical panel results do not guarantee future provider behavior.
- Grading exclusions and limited repeated runs affect the harness comparison.

See [methodology](methodology.md) for repeated-call controls and [retractions](retractions.md) for withdrawn claims.
The [scope record](proposals/ari-decomposition.md) distinguishes embedding, decoding, and harness measurements.

## Evidence gaps

Additional float detectors require retained vectors. The original panel did not retain them; the later SFR capture did.
See [raw evidence availability](analysis/float-detector-gate.md).

The approximately 14-fold LSH sensitivity comparison remains a single synthetic result.
It requires replication on real model outputs.
Controlled harness experiments also require more repeated runs and accessible grading evidence.
