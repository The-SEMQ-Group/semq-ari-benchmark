# ARI-E power: what harness effect could the metric actually detect?

**Status: run (simulation).** Script: [`power.py`](power.py). Machine-readable:
[`results/power.json`](results/power.json) (attested; sidecars alongside).
Reproduce: `python power.py` — deterministic (seed recorded in the JSON),
200 simulated experiments per cell over a (gap × repeats × cases) grid,
base pass rate 0.75.

The question, asked before spending a day of agent compute: with a known
harness effect planted, how often does the ARI-E estimator's interval on the
effect exclude zero?

## Headline numbers

| true gap | repeats | cases | power |
| --- | --- | --- | --- |
| 0.20 | 2 | 25 | **0.075** |
| 0.20 | 5 | 50 | 0.46 |
| 0.20 | 5 | 100 | 0.74 |
| 0.20 | 5 | 200 | **0.965** |
| 0.10 | 5 | 200 | 0.175 |

- **The comparison that motivated this project would have failed silently.**
  At octobench's design point — 25 cases, 2 repeats — a 20-point pass-rate
  gap is detected 7.5% of the time. A null read at that size says almost
  nothing.
- **Repeats and cases both matter, and neither alone is enough.** 5 repeats
  at 50 cases reaches only 0.46 for a 20-point gap; comfortable power needs
  100–200 cases *with* the repeats.
- **A 10-point gap is out of reach at every size tested** (max 0.175 at
  200×5). Claims at that effect size need a different design, not more of
  this one.
- **Small case counts also lie in the other direction**: with no true effect,
  the false-positive rate reaches **0.18 at 10 cases** — intervals misbehave
  below ~25 cases, so a "detected" effect from a tiny suite is as suspect as
  a null from one.

## How to read this against the measured ARI-E result

The Open-SWE measurement (`../harness-effect/`) satisfies these requirements
by orders of magnitude (~11,000 cases per contrast, up to 3 rollouts), which
is why its intervals exclude zero cleanly. This simulation is the reason the
project did not instead run a 25-case suite and publish whatever came out.
