# Availability of raw embedding evidence

## Original panel

The original 13-model panel did not retain the raw embedding matrices needed for additional float detectors.
Its artifacts contain aggregate measurements, quantized codes, or code digests.
A code digest cannot recover the original vectors.
The float-exact, cosine, and top-k detectors therefore remain unmeasured for those historical captures.

## Later capture

The [Salesforce/SFR-Embedding-2_R capture](../../experiments/deployed-agent-panel/sfr_capture.py) retains raw matrices for each condition.
See the [panel results](../../experiments/deployed-agent-panel/RESULTS.md) for its scope.
This later evidence does not recover raw vectors from the original panel.

## Requirements for another detector

1. Retain the baseline and condition vectors before quantization.
2. Bind the arrays to input order, model revision, calibration, and environment metadata.
3. Define the detector's unit, reference storage, and comparison rule.
4. Evaluate the detector on the retained vectors.
5. Report uncertainty and the repeated-call baseline.

Re-running a historical API capture produces a new measurement.
It cannot establish the original float differences because provider responses and environments can change.
A local reproduction also requires the original model and environment pins.

The [report schema](../../spec/report-schema.json) supports named detectors under each condition.
Schema support alone does not establish that a detector has been implemented or measured.
