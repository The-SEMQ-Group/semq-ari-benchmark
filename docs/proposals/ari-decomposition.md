# Measurement scope: representation and decoding

This document records the measurement scopes proposed in August 2026.
A third scope, ARI-E (harness effects on agent outcomes), was proposed here and later removed; see the [changelog](../../CHANGELOG.md).
The implemented specifications and report schema define current behavior.
This record does not rename schema fields or create a combined score.

| Scope | Measurement | Implementation and evidence |
| --- | --- | --- |
| ARI-R | Code agreement for model representations under deployment changes. | [Embedding specification](../../spec/ari-canonical-v0.1.md) and [panel results](../../experiments/deployed-agent-panel/RESULTS.md). |
| ARI-D | Agreement of generated output under repeated requests and changed conditions. | [Decoding specification](../../spec/arid-bench-v0.1.md) and [API dry run](../../experiments/arid-dry-run/RESULTS.md). |

## Representation scope

The embedding schema retains the `ARI` field.
Its value averages HER over the present core conditions: `proc`, `conc`, and `time`.
It does not measure complete agent behavior.

## Decoding scope

The initial internal-logit experiments compare logits and generated tokens.
Their Hamming measurements are distinct from the public-output detectors in ARI-D-Bench-v0.1.
Use the detector name, unit, and reference condition when reporting either measurement.
A changed logit code does not imply a changed output token.

## Remaining work

Keep the two scopes separate in comparisons.
A combined report format or aggregate score requires a specification change.
See [findings](../findings.md) for current evidence limits.
