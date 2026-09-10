# Decoding detector comparison

The comparison uses one CPU model, 539 reference steps, and five noise trials.
Data: [baselines.json](results/baselines.json). Implementation: [baselines.py](baselines.py).
Gaussian perturbation size is denoted by σ.

## Response to perturbation

Different statistics use different units. A response ratio is not a calibrated detection-power comparison.

| statistic | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1.0 | order |
| --- | ---: | ---: | ---: | ---: | ---: | :--- |
| **SEMQ H̄** | 1.33e-4 | 1.33e-3 | 1.33e-2 | 0.129 | 0.813 | **linear** |
| top-2 margin Δ | 1.14e-4 | 1.15e-3 | 1.14e-2 | 0.112 | 0.873 | **linear** |
| max \|Δlogit\| | 4.29e-4 | 4.28e-3 | 4.28e-2 | 0.429 | 4.28 | linear |
| top-20 set change | 1.77e-4 | 7.07e-4 | 5.62e-3 | 0.051 | 0.346 | sub-linear |
| KL | −5.1e-10 | 2.18e-7 | 2.23e-5 | 2.21e-3 | 0.226 | **quadratic** |
| JS | −1.9e-10 | 5.39e-8 | 5.57e-6 | 5.50e-4 | 0.051 | quadratic |
| token flip | 0 | 1.86e-3 | 2.97e-3 | 0.029 | 0.236 | threshold |

## Reference storage

Storage is per step for a vocabulary of 32,000. It excludes computation and metadata costs.

| statistic | bytes |
| --- | ---: |
| token flip | 4 |
| **top-2 margin Δ** | **8** |
| top-20 set change | 80 |
| **SEMQ H̄** | **16,000** |
| KL / JS / max\|Δlogit\| / L2 | 128,000 |

## Arithmetic reassociation

The recorded comparison changes algebraic grouping or summation order. It does not measure every platform or implementation.

| statistic | bit-identical | max relative difference |
| --- | :---: | ---: |
| **SEMQ H̄** | **yes** | 0 |
| top-2 margin Δ | yes | 0 |
| max \|Δlogit\| | yes | 0 |
| L2 \|Δlogit\| | no | 1.4e-7 |
| **KL** | **no** | **4.6%** |
| **JS** | **no** | **363%** |
| top-20 set change | no | 100% |
| token flip | no | 100% |

## Perturbations below selected ranks

This synthetic test holds the top ranks fixed and applies noise at σ = 0.1.

| statistic | full vector | below rank 20 | below rank 100 |
| --- | ---: | ---: | ---: |
| **SEMQ H̄** | 0.1288 | **0.1289** | **0.1285** |
| max \|Δlogit\| | 0.4289 | 0.4283 | 0.4281 |
| KL | 2.21e-3 | 2.19e-4 | 6.12e-5 |
| top-20 set change | 0.0509 | 0.0335 | **0** |
| **top-2 margin Δ** | 0.1115 | **0** | **0** |
| **token flip** | 0.0289 | **0** | **0** |

## Interpretation

At σ = 1e-3, the margin change was 1.15e-3 and SEMQ H̄ was 1.33e-3.
The margin required eight reference bytes per step; the recorded SEMQ state required 16,000.
The numerical response ratio was approximately 86 percent.

KL and JS returned small negative values at σ = 1e-4 because of numerical error.
Their values also changed under the tested arithmetic regrouping.
This limits exact recomputation of those implementations; it does not rule out threshold-based detection with KL or JS.

Under synthetic noise below rank 20, the top-2 margin and token-flip statistic remained zero.
SEMQ still changed. This demonstrates broader vector coverage in that constructed test.
See [rank-profile results](../drift-rank-profile/RESULTS.md) for adapter and token-bias experiments.

## Limits

These results do not establish a universally best detector.
The synthetic noise is not a measurement of deployment-change frequency.
Storage, response magnitude, reproducibility, and coverage answer different questions.

## Reproduce

Install the [SEMQ SDK](../../ari/README.md#install-the-canonical-probe) and model dependencies.
Generate the matching reference cache with `run_matrix.py` before running the baseline comparison.
From the repository root:

```bash
python experiments/decoding-reproducibility/baselines.py
```

The command writes `experiments/decoding-reproducibility/results/baselines.json`.
Check the source's reference-cache requirements before comparing results from another model or prompt set.
