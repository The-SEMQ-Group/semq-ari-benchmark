# Hosted-decoding power simulation

The simulation contains 252 cells with 200 trials per cell and 500 bootstrap resamples per arm.
Data: [power_grid.json](results/power_grid.json). Implementation: [run_sim.py](run_sim.py).
Each cell has a fixed seed.

The grid varies baseline agreement, repeat count, effect size, and deviation distribution.
Bimodal deviations share one alternative output. Unique deviations each produce a different output.
Effects are either uniform or concentrated across prompts.

## Results at 100 prompts

The protocol uses eight repeats, with twelve for concurrency. The Δ=0 column reports false-positive rates.

| floor | deviations | effect shape | Δ=0 | Δ=0.05 | Δ=0.10 | Δ=0.20 |
|---|---|---|---:|---:|---:|---:|
| 1.0 | any | any | 0.00 | 1.00 | 1.00 | 1.00 |
| 0.9 | bimodal | uniform | 0.00 | 0.23 | **0.96** | 1.00 |
| 0.9 | bimodal | concentrated | 0.00 | 0.20 | **0.91** | 1.00 |
| 0.9 | unique | uniform | 0.00 | 0.09 | 0.77 | 1.00 |
| 0.9 | unique | concentrated | 0.00 | 0.04 | **0.53** | 1.00 |
| 0.7 | bimodal | uniform | 0.01 | 0.26 | **0.90** | 1.00 |
| 0.7 | bimodal | concentrated | 0.01 | 0.25 | **0.89** | 1.00 |
| 0.7 | unique | uniform | 0.01 | 0.17 | 0.64 | 1.00 |
| 0.7 | unique | concentrated | 0.01 | 0.11 | **0.55** | 1.00 |

## Interpretation

At noisy baselines, a 0.10 decrease had power 0.89–0.96 under bimodal deviations.
Concentrated unique deviations reduced power to approximately 0.53–0.55.
Increasing repeats from eight to twelve did not resolve that case; increasing prompt count is the relevant design change.
Five-point decreases had low power at noisy baselines.
False-positive rates were at most one percent in this grid.

A classification of `≈ same` does not exclude a concentrated 0.10 decrease.
The CI-overlap rule trades lower false-positive rates for lower detection power.
These conclusions depend on the simulated prompt-stability distributions.
The simulation does not model a backend that changes within a capture arm.

## Reproduce

After the root development setup, run:

```bash
python experiments/arid-power-sim/run_sim.py
```

The command writes `experiments/arid-power-sim/results/power_grid.json`.
Compare the random seeds, grid, and trial counts before comparing outputs.
