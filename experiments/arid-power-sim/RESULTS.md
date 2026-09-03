# ARI-D power simulation. Does the frozen design see the effects we claim to care about?

**Status: run, deterministic.** Script: [`run_sim.py`](run_sim.py). Machine-readable:
[`results/power_grid.json`](results/power_grid.json). 252 cells, 200 simulated panel runs
per cell, 500-resample bootstrap per arm, seeded per cell — two runs produce identical
JSON. Runtime ~3 s.

**Headline: the frozen design (100 prompts, k=8/12) detects the effect the spec names —
a 0.10 drop in `exact_generation` — with ≥0.89 power under bimodal deviations at every
floor tested, and the conservative CI-overlap rule holds false positives at ≤1%. But the
sim also found a blind spot the spec must state: when the effect concentrates on few
prompts AND deviations compound uniquely, power at Δ=0.10 is ~0.53 — and adding repeats
does not fix it, only adding prompts does.**

## What was simulated

The scorer's own statistics, end to end: per-prompt pair-identical share over C(k,2)
repeat pairs, arm value = mean over prompts, 95% percentile bootstrap over prompts,
condition flagged only when its CI and `same`'s CI are disjoint (the Matrix's
conservative overlap rule). Ground truth is exact: prompt stabilities are solved
numerically so the true mean pair rate drops by exactly the target Δ.

Axes: floor (`same`'s own mean pair rate) ∈ {1.0, 0.9, 0.7} · k ∈ {4, 8, 12} ·
Δ ∈ {0, 0.05, 0.10, 0.20} · deviation shape (bimodal = one alternative completion,
pair rate bounded below by 0.5 · unique = every deviation its own, bound 0) · effect
shape (uniform across prompts · concentrated on the low-margin tail, widening as
needed). The fragile-tail width adapts to the floor — a provider with a low `same`
floor is unstable on more prompts, not infinitely unstable on a fixed few.

## Results at the frozen design (n=100, k=8; k=12 for `conc`)

Power (share of 200 trials flagged), FPR in the Δ=0 column:

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

## Reading

**1. The design's promises hold where the spec makes them.** Δ=0.20 is seen always and
everywhere (1.00). Δ=0.10 is seen with 0.89–0.96 power under bimodal deviations at noisy
floors, and trivially at a clean floor. False positives from the conservative overlap
rule: ≤1% across all 252 cells. k=4 would not have been enough (0.44–0.74 at Δ=0.10);
k=8 is justified.

**2. The blind spot, stated plainly.** Unique-compounding deviations concentrated on the
low-margin tail at Δ=0.10: power ~0.53 at every floor — a coin flip. The mechanism is
visible in the sim: the limiting variance is *across prompts* (20–30 prompts carry the
whole effect, so the bootstrap-over-prompts CI stays wide), not within-prompt.

**3. That mechanism bounds which lever works.** Raising k from 8 to 12 moves
unique-concentrated power not at all (0.53 → 0.53 at floor 0.9); halving prompts to 50
collapses it (0.53 → 0.09). The harness-power lesson — repeats sharpen the control more
than cases — is true for within-prompt noise and inverts in the cross-prompt-concentrated
regime: there, only prompts help. The frozen k=8/12 stands; the case count is the
binding resource, and 100 is the working minimum, not a comfortable ceiling.

**4. Δ=0.05 is not a claim this bench can make.** Power ≤0.33 at any noisy floor. The
spec should continue to name 0.10 as the effect size of interest and stay silent on
five-point drops.

**5. What this means for the classification, honestly.** The conservative CI-overlap
rule buys its ≤1% FPR at the price of point 2. A missed flag reads `≈ same` — the bench
under-claims rather than over-claims, which is the right failure direction for an
instrument that names companies. But section 8's language should carry the caveat: a
`≈ same` reading on `exact_generation` does not exclude a concentrated drop of ~0.10
under compounding deviations; `first_divergence` (graded, per-pair) is the detector with
a shot at those, which is one more reason the graded diagnostics ship beside the
headline. Whether v0.2 should add a paired per-prompt test for this regime is a
methodology decision, flagged here with the evidence rather than decided.

## Limits

The generative model is a two-population stability mixture; real prompt-stability
distributions are unmeasured until the dry run. Deviation reality sits between the
bimodal and unique brackets — both are reported, neither is claimed. The `time` arm is
simulated as same-run (no drift-within-arm); a drifting backend mid-run would look like
extra within-arm variance and lower power further, which the dry run can bound.
