# Experiments

Start with the [development setup](../README.md#build-and-test).
Run commands from the repository root unless a procedure specifies another directory.
Experiment scripts can overwrite their result files. Use a separate checkout to reproduce a published run.

| Experiment | Purpose | Procedure and results |
| --- | --- | --- |
| Drift sensitivity | Response to synthetic embedding perturbations. | [Procedure](drift-sensitivity/README.md), [results](drift-sensitivity/RESULTS.md). |
| Deployed embedding panel | Reproducibility across model deployment conditions. | [Procedure](deployed-agent-panel/README.md), [results](deployed-agent-panel/RESULTS.md). |
| GPU determinism | Effects of precision and GPU configuration. | [Procedure](gpu-determinism/README.md), [results](gpu-determinism/RESULTS.md). |
| Regime discrimination | Code changes compared with retrieval quality. | [Results and reproduction](regime-discrimination/RESULTS.md). |
| Internal decoding | Logit and output changes under serving conditions. | [Initial results](decoding-reproducibility/RESULTS.md), [larger models](decoding-reproducibility/RESULTS-production-models.md). |
| Decoding baselines | Sensitivity and storage for alternative detectors. | [Baseline comparison](decoding-reproducibility/BASELINES.md). |
| Drift rank profile | Distribution of changes across logit ranks. | [Results](drift-rank-profile/RESULTS.md). |
| Probe verifiability | Reproducibility of frozen vector quantizers. | [Results](probe-verifiability/RESULTS.md). |
| Probe sweep | Bin count and calibration, against storage and saturation. | [Procedure and reproduction](probe-sweep/README.md). |
| ARI-D power | Statistical power of the hosted-output protocol. | [Simulation](arid-power-sim/RESULTS.md). |
| ARI-D dry run | Hosted API captures and protocol calibration. | [Procedure](arid-dry-run/README.md), [results](arid-dry-run/RESULTS.md). |
| ARI-D references | Output from published weights at fixed precision. | [Procedure](arid-fp32-refs/README.md). |
| Harness effect | Within-harness and cross-harness agreement. | [Results](harness-effect/RESULTS.md). |
| Harness power | Case counts and repeated runs for harness comparisons. | [Simulation](harness-power/RESULTS.md). |

## Dependencies

The two power simulations use the base package dependencies.
Model experiments require their model libraries, data, and sometimes the separate SEMQ SDK.
Use the [harness guide](../ari/README.md) for SDK access and [infrastructure guide](../infra/README.md) for GPU runs.
Hosted captures require provider credentials and incur API charges.
Published JSON and CSV files can be inspected without rerunning captures.

## Report a reproduction

Record the commit, command, model revision, dependency versions, hardware, and output hashes.
State which conditions ran and which were skipped.
Compare outputs only when the protocol and metric units match.
Keep new measurements separate from the frozen reference artifacts until review.
