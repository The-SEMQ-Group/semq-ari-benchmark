# Harness-effect power simulation

**Status: rerun on 2026-09-16 with the current estimator in [ari/harness.py](../../ari/harness.py).**
The result file records the SHA-256 of that estimator file. The output of 2026-08-07 is kept in [results/superseded-2026-08-07/](results/superseded-2026-08-07/) with its attestation.

The simulation measures how often the ARI-E interval excludes zero when an effect is present.
It uses 200 simulated experiments per cell, a baseline pass probability of 0.75, 400 bootstrap resamples, and seed 0.
Two harnesses get per-case pass probabilities that differ by the gap. Outcomes are independent Bernoulli draws.
The true effect is the mean of the squared gap over cases, so a gap of 0.20 is an effect of 0.04.
Data: [power.json](results/power.json). Implementation: [power.py](power.py).

The grid covers equal repeats (2/2, 3/3, 5/5) and unequal repeats (2/3, 2/5), the shape the Open-SWE-Traces contrasts have.

## False-positive rate

Share of trials whose interval excludes zero when the true gap is zero. The nominal rate is 5 percent.
With 200 trials per cell, one trial is 0.5 percent and the Monte Carlo standard error at 5 percent is about 1.5 percent.

| repeats | 10 cases | 25 | 50 | 100 | 200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2/2 | 0.040 | 0.080 | 0.065 | 0.070 | 0.040 |
| 3/3 | 0.150 | 0.085 | 0.110 | 0.105 | 0.055 |
| 5/5 | 0.125 | 0.135 | 0.080 | 0.045 | 0.060 |
| 2/3 | 0.110 | 0.100 | 0.080 | 0.075 | 0.055 |
| 2/5 | 0.100 | 0.105 | 0.105 | 0.050 | 0.055 |

Below 100 cases the rate exceeds the nominal 5 percent in most cells, up to 0.15 at ten cases with 3/3 repeats.
At 200 cases every cell is within 0.06. Unequal repeats do not raise the rate beyond the equal-repeat cells.

## Power

| true gap | true effect | repeats | cases | power |
| --- | --- | --- | --- | --- |
| 0.20 | 0.040 | 2/2 | 25 | **0.065** |
| 0.20 | 0.040 | 5/5 | 50 | 0.44 |
| 0.20 | 0.040 | 5/5 | 100 | 0.79 |
| 0.20 | 0.040 | 5/5 | 200 | **0.95** |
| 0.20 | 0.040 | 2/3 | 200 | 0.44 |
| 0.20 | 0.040 | 2/5 | 200 | 0.545 |
| 0.10 | 0.010 | 5/5 | 200 | 0.18 |
| 0.40 | 0.160 | 2/2 | 50 | 0.70 |
| 0.40 | 0.160 | 2/3 | 50 | 0.81 |

The full grid is in `power.json`.

## Interpretation

At 25 cases and two repeats, the design detected a 0.20 gap in 6.5 percent of trials, which is the false-positive rate.
At 200 cases and five repeats, power reached 0.95 for that gap.
For a 0.10 gap, the maximum measured power was 0.18. A 0.05 gap is not detectable in this grid.
Adding a third repeat on one side (2/3) raises power over 2/2 at every case count; a fifth repeat on one side (2/5) helps less than a third repeat on both sides (3/3).
These results do not support small-effect claims from a small case set.

The [Open-SWE-Traces analysis](../harness-effect/RESULTS.md) uses 3,779 to 12,195 cases with two or three repeats per side, outside this grid.
Its permutation null gives the false-positive check at that scale. Its data collection assumptions remain separate from this simulation.

## Comparison with the superseded output

The 2026-08-07 output reported 0.075, 0.46, 0.74, 0.965, and 0.175 for the first five rows above, and a false-positive rate of 0.18 at ten cases.
The rerun gives 0.065, 0.44, 0.79, 0.95, and 0.18, and 0.15 at ten cases with 3/3 repeats.
The differences are within the Monte Carlo error of 200 trials. The two runs draw different random streams because the grid changed.

## Reproduce

After the root development setup, run:

```bash
python experiments/harness-power/power.py
```

The command writes `experiments/harness-power/results/power.json` in a few minutes on one CPU.
Compare the seed, grid, trial count, and `harness_sha256` before comparing the reported power.
