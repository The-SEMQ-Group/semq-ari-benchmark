# Panel methodology and controls

HER measures agreement of embedding codes for identical inputs.
A changed code means the embedding crossed a quantizer boundary. It does not establish a semantic or answer change.
Use the [panel results](../experiments/deployed-agent-panel/RESULTS.md) for canonical measurements.

## OpenAI pilot

The illustrative pilot used 1,000 sample inputs and three repeats per condition.
It used `text-embedding-3-large`, 3,072 dimensions, float output, one input per request, and SEMQ 1.2.0.
Calibration used QBIN n=2 at the 99th percentile.
The input sample differed from the frozen benchmark.

| Condition | HER | 95% bootstrap interval |
| --- | ---: | --- |
| Persistent client (`same`) | 0.852 | [0.834, 0.870] |
| Fresh client (`proc`) | 0.851 | [0.833, 0.868] |

Evidence: [pilot JSON](../experiments/deployed-agent-panel/artifacts/openai_proc_pilot_n1000.json).
The intervals overlap. This comparison did not distinguish fresh-client variability from persistent-client variability.
It does not identify the provider's routing mechanism.

## Local model control

The local pilot used `BAAI/bge-large-en-v1.5` and 64 inputs.
Every tested configuration produced `same = proc = 1.000` with zero Hamming distance.

| Configuration | Evidence |
| --- | --- |
| Apple Silicon, pinned threads | [Capture](../experiments/deployed-agent-panel/artifacts/selfhosted_bge_large_pinned.json). |
| x86, OpenBLAS 0.3.27, four PyTorch threads | [Capture](../experiments/deployed-agent-panel/artifacts/selfhosted_bge_large_x86_unpinned.json). |
| x86, one thread | [Capture](../experiments/deployed-agent-panel/artifacts/selfhosted_bge_large_x86_pinned.json). |

This pilot did not reproduce the historical cross-process BLAS incident.
A later shared-instance concurrency test produced an approximately 0.1-percent input flip on bge-large.
The panel's reported concurrency configuration used batched or replica execution instead.
Do not generalize its result to every concurrency implementation.

## Float checks

Four immediate repeats of one text returned identical vectors.
In two separated passes over 30 texts, three texts returned different vectors.
Approximately 2,300 of 3,072 dimensions changed in those vectors, with differences around `1e-4`.
These observations support an embedding-level difference. They do not establish a cache or replica mechanism.
The original raw matrices were not retained; see [evidence availability](analysis/float-detector-gate.md).

## Gemini cache checks

A test appended unique nonces to 150 inputs.
Gemini produced ARI 1.000; the OpenAI control produced ARI 0.807 on the same inputs.
Gemini also retained agreement across the panel's approximately 76-hour comparison.

These checks reduce the common-string-cache explanation.
They do not prove that the provider uses deterministic computation or exclude all caching strategies.
The supported result is repeated output agreement under the measured conditions.

## Interpretation

Treat `same` as a repeated-call measurement for APIs.
For deterministic local inputs, test the probe separately before attributing code changes to the model.
Compare each condition with its own repeated-call baseline.
Report uncertainty, input scope, environment, and elapsed time.
See [null-relative analysis](analysis/null-relative-effects.md) for the limits of baseline subtraction.
