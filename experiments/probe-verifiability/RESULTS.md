# Probe verifiability. How exposed is a frozen VQ to reassociation?

**Status: run.** Real corpus, real k-means codebooks, five configurations.
Script: [`near_tie_sweep.py`](near_tie_sweep.py). Machine-readable:
[`results/near_tie_sweep.json`](results/near_tie_sweep.json).

**Headline: the exposure is far smaller than an earlier synthetic estimate, and the
discrete-attractor property does not break on any real codebook tested.** An earlier
version of the probe-choice analysis claimed otherwise ([retraction ledger](../../docs/retractions.md)). This experiment is what corrected it.

## Setup

5,000 SciFact abstracts, embedded with `all-MiniLM-L6-v2` (384-d, L2-normalised), split
half for fitting and half for probing. Codebooks are fitted by k-means, 256 centroids per
subspace (8 bits, the standard setting), seed 0. The sweep varies only the number of
subspaces, which sets how many dimensions each centroid has to separate in.

## Results

| subspaces | dims/subvector | near-tie <1e-6 | near-tie <1e-5 | median margin | forms disagree (symbols) | forms disagree (rows) | `encode(reconstruct(c)) = c` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 32 | 12 | 0.0450% | 0.3550% | 1.87e-03 | 0.0000% | 0.0000% | holds, both forms |
| 64 | 6 | 0.0700% | 0.7444% | 8.68e-04 | 0.0006% | 0.0400% | holds, both forms |
| 128 | 3 | 0.3044% | 3.0791% | 2.05e-04 | 0.0000% | 0.0000% | holds, both forms |
| 192 | 2 | 2.6206% | 13.7535% | 4.50e-05 | 0.0000% | 0.0000% | holds, both forms |
| 384 | 1 | 91.1873% | 99.5359% | 3.22e-07 | 0.0122% | 4.6400% | holds, both forms |

At the standard 32×256 setting the minimum margin over all assignments is 7.45e-09, so
near-ties do exist. They are simply rare. At the 1-dimensional setting the minimum margin
is exactly 0: with 256 centroids on a line, k-means returns duplicates.

**Run-to-run variation.** `KMeans(n_init=4)` is not bit-reproducible across runs here, so
these figures move slightly. A second run of the same script gave 0.3041% / 3.0787% at 128
subspaces and 0.0115% / 4.36% at 384. Read the last row as "roughly 4–5% of rows," not as
4.64%. The qualitative result is identical in both runs: the attractor property holds
everywhere, and the disagreement stays small.

## Reading

**1. Near-tie density is real and strongly configuration-dependent.** It rises by three
orders of magnitude across the sweep, from 0.045% at the standard setting to 91% when each
centroid has one dimension to work with. The mechanism behind the concern is genuine:
crowding centroids does produce near-ties, exactly as predicted.

**2. Near-ties do not, on their own, break the probe.** This is the finding that matters,
and it is the opposite of what was predicted. Even at 91% near-tie density with a median
margin of 3.2e-07, `encode(reconstruct(c)) = c` holds under **both** distance formulations,
at every setting. The reason is structural: `reconstruct(c)` returns the centroid itself,
and a centroid's distance to itself is zero to within rounding, which stays below the
distance to any distinct centroid even when that distance is small.

**3. Formulation disagreement is real but rare.** Direct and expanded forms disagree on at
most 0.012% of symbols. In row terms, a row being one probed vector, that is 0% at most
settings, 0.04% at 64 subspaces, and 4–5% at the degenerate 1-dimensional setting. So
roughly 1 vector in 2,500 can read differently at a plausible configuration, depending only
on which algebraic identity the library implementing the probe happened to choose. That is
a genuine verifiability defect. It is not a large one.

## What this retracts

An earlier version of the probe-choice analysis reported that the two formulations disagree
on **39% of symbols and 100% of rows**, and that `encode(reconstruct(c)) = c` **fails**
under the expanded form. Those
numbers are correct for the codebook they were measured on, which is the constructed
codebook in the SDK's `tests/integration/test_probe_arch_invariance.py`: it pairs every
centroid with a near-duplicate placed 1e-6 away, deliberately.

**k-means does not produce that geometry.** Two centroids 1e-6 apart do not survive a
k-means fit. They compete for the same Voronoi mass and merge. The constructed codebook
shows that the mechanism *can* break a VQ probe. It says nothing about whether a
shipped one *is* broken. The error was to generalize from it to "real codebooks are the
latter".

The 1.6% near-tie figure that the earlier text called "typical" also came from synthetic
data with 2-dimensional subvectors. Measured properly, 2-dimensional subvectors give 2.62%.
The estimate was about right *for that configuration*. That configuration is not typical,
and the standard one is 58× lower.

## What survives

A frozen VQ probe's invariance is **contingent and must be measured per codebook**: the
sweep shows it varies from clean to 4.36% of rows across configurations of the same
algorithm on the same data, and nothing in a shipped codebook file announces which regime
it is in. SEMQ's invariance follows from the operator's form: threshold comparisons
against a calibrated scale, with no pair of candidates to reorder. A verifier can check it
in advance instead of auditing it after the fact.

That is a real distinction and it is the one
[probe-validation](../../docs/analysis/probe-validation.md) now makes. It is considerably weaker
than "a frozen VQ is not verifiable," which is what the earlier text said and what this
experiment does not support.

## Caveats

- One encoder, one corpus, one seed. Near-tie density depends on the data distribution.
  A corpus with many duplicates would crowd the centroids more.
- The sweep simulates reassociation by computing the same distance two algebraically
  equivalent ways. That is a proxy for what differing hardware does to the arithmetic, not
  a measurement across actual architectures. The SDK test asserts the real thing on three
  platforms per commit.
- The 1-dimensional setting is included to bracket the trend, not because anyone ships it.
