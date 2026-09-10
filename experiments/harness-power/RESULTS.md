# Harness-effect power simulation

The simulation measures how often the ARI-E interval excludes zero when an effect is present.
It uses 200 simulated experiments per cell and a baseline pass probability of 0.75.
The result JSON records the random seed.
Data: [power.json](results/power.json). Implementation: [power.py](power.py).

## Results

The gap is a pass-probability difference. Power is the fraction of trials that detect the effect.

| true gap | repeats | cases | power |
| --- | --- | --- | --- |
| 0.20 | 2 | 25 | **0.075** |
| 0.20 | 5 | 50 | 0.46 |
| 0.20 | 5 | 100 | 0.74 |
| 0.20 | 5 | 200 | **0.965** |
| 0.10 | 5 | 200 | 0.175 |

## Interpretation

At 25 cases and two repeats, the design detected a 0.20 gap in 7.5 percent of trials.
At 200 cases and five repeats, power reached 0.965 for that gap.
For a 0.10 gap, the maximum measured power was 0.175.
With no effect and ten cases, the false-positive rate reached 0.18.
These results do not support small-effect claims from a small case set.

The [public harness analysis](../harness-effect/RESULTS.md) uses substantially more cases.
Its data collection assumptions remain separate from this simulation.

## Reproduce

After the root development setup, run:

```bash
python experiments/harness-power/power.py
```

The command writes `experiments/harness-power/results/power.json`.
Compare the seed, grid, and trial count before comparing the reported power.
