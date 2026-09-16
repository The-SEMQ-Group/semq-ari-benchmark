# Measurement scope: representation, decoding, and harness effects

This document records the three measurement scopes proposed in August 2026.
The implemented specifications and report schema define current behavior.
This record does not rename schema fields or create a combined score.

| Scope | Measurement | Implementation and evidence |
| --- | --- | --- |
| ARI-R | Code agreement for model representations under deployment changes. | [Embedding specification](../../spec/ari-canonical-v0.1.md) and [panel results](../../experiments/deployed-agent-panel/RESULTS.md). |
| ARI-D | Agreement of generated output under repeated requests and changed conditions. | [Decoding specification](../../spec/arid-bench-v0.1.md) and [API dry run](../../experiments/arid-dry-run/RESULTS.md). |
| ARI-E | Difference between within-harness and cross-harness outcome agreement. | [Harness implementation](../../ari/harness.py) and [results](../../experiments/harness-effect/RESULTS.md). |

## Representation scope

The embedding schema retains the `ARI` field.
Its value averages HER over the present core conditions: `proc`, `conc`, and `time`.
It does not measure complete agent behavior.

## Decoding scope

The initial internal-logit experiments compare logits and generated tokens.
Their Hamming measurements are distinct from the public-output detectors in ARI-D-Bench-v0.1.
Use the detector name, unit, and reference condition when reporting either measurement.
A changed logit code does not imply a changed output token.

## Harness scope

ARI-E adjusts cross-harness agreement by measured self-consistency.
It requires repeated runs per case and harness.
The implementation accepts trajectories in JSON Lines format and does not require model-state checkpoints or `semq.kv`.
The public dataset result does not establish that harness effects dominate for every model or task.

## Remaining work

Keep the three scopes separate in comparisons.
A combined report format or aggregate score requires a specification change.
Controlled harness comparisons need adequate case counts, repeated runs, and auditable grading.
See [findings](../findings.md) for current evidence limits.
