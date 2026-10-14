# ARI-Canonical-v0.2

Status: draft. Supersedes [ARI-Canonical-v0.1](ari-canonical-v0.1.md) once the measurements below are regenerated.
This specification fixes the embedding probe for ARI v0.2.
It changes the probe, so v0.1 and v0.2 codes are not comparable.

## Probe requirements

Use SEMQ QUANT from the public SEMQ SDK, `semq` 1.x, with `n_bins = 2`: four symbols, two bits per dimension.
The range is fixed by the dimension alone: `s = (float)(2 / sqrt(dim))`, the SDK's `CodecConfig.max_magnitude`.
There is no calibration and no reference distribution.

Before encoding, rescale every row to unit L2 norm: compute the norm in binary64, divide, and cast to float32.
The SDK rejects rows whose squared norm is more than `2^-10` from 1; the rescaling makes provider norm conventions irrelevant.
A zero row has no direction and is an error.

The probe must return identical codes for identical vectors.
The SDK pins its bytes with conformance vectors on every supported CPU and binding.
Record the SDK version and build id with every capture.

## Changes from v0.1

| | v0.1 | v0.2 |
| --- | --- | --- |
| SDK | private builds, 1.2–1.5 | public `semq` 1.x, PolyForm Noncommercial 1.0.0 |
| Range `s` | 99th percentile of a reference set, per model | `2 / sqrt(dim)`, the same for every model of a dimension |
| Input | vectors as returned | rows rescaled to unit norm |
| Symbol and packing | `sign * bins + bin`, least significant bit first | unchanged |

`fingerprint.s` in the report schema keeps its name and now records `2 / sqrt(dim)`.
The harness refuses to encode against a stored v0.1 baseline scale.

## Design rationale

A calibrated range depends on the corpus that produced it.
On the frozen inputs, calibrating on the FiQA rows instead of the SciFact rows moves `s` by 0.3–3.6%.
For the same vectors, that changes the `n_bins = 2` code of 60–100% of rows across eight encoders.
A fixed range removes that dependence, and anyone with the public SDK computes the same codes from the same vectors.
The [port proposal](../docs/proposals/public-sdk-port.md) gives the comparison against ground truth.

## Known limits

The sign boundary is unchanged from v0.1.
Coordinates at or near zero flip sign under any additive perturbation.
`all-MiniLM-L6-v2` and `all-mpnet-base-v2` carry such coordinates, which explains the v0.1 perturbation floor for those models; it applies to v0.2 equally.

A fixed range does not adapt to a model whose coordinate distribution is far from isotropic.
None of the eight measured encoders shows this: 3–5% of their coordinates reach the range limit.
Report the saturated fraction when adding a model.

A boundary that falls exactly on a value the encoder can emit makes codes fragile to rounding.
At `dim = 1024` the magnitude boundary is `1/32`, which lies on the float16 grid.
Encode float32 outputs, and record when a provider returns lower precision.

## Fingerprint terms

The v0.1 registry values of `b` and `κ` were fitted with the calibrated probe.
They must be refitted with this probe before v0.2 is frozen.
`s` is no longer a fitted term.

## Registry maintenance

Publish the model identifier, dimension, SDK version, fit parameters, and coverage status.
Changes to frozen probe parameters require a new specification version.
