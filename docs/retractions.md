# Retractions — what the data support, and what was withdrawn

This project retracts claims in public when its own experiments refute them,
with the reasoning attached. This file is the standing ledger; the summary
table lives in [findings.md](findings.md) §9. (This record originally lived
in the project whitepaper, since removed in favor of the peer-reviewed paper
— *ARI: Measuring Reproducibility in Deployed ML Systems*, NeurIPS 2026
Workshop on Machine Learning for Systems.)

An earlier framing led with *"SEMQ is ~300× more sensitive than retrieval."* The
fine-grained drift-sensitivity sweep — a wider model grid with bootstrap CIs and
pre-registered hypotheses — **does not support that number**, and the reason is instructive:
two of the three pre-registered gates were mis-designed for the phenomenon.

**Sensitivity-gap gate — `SEMQ slope ÷ recall slope ≥ 100×` → refuted.** It is a *ratio of
two slopes*. On a clean corpus Recall@10 is degenerate: pinned at 1.0 for some models
(ratio → ∞), barely moving for others (ratio → ~1). The gate measures recall's degeneracy,
not SEMQ's sensitivity. Per-model ratios came out 1.1–4.2 and ∞ — noise, not signal.

**Theory-match gate — `(b, a)` within 5% of `(1, √(2·dim/π))` → refuted only on the
prefactor.** The slope matches (max error 0.068; 5/6 within 4%). The gate fails *only*
because the empirical prefactor is 1.48–3.06× the uniform-on-sphere value (across the full
13-model registry) — and that "failure" **is the κ discovery**. The gate refutes; the science
generalises.

**Universality gate — `std(b) ≤ 0.05` → confirmed** (std = 0.028 across the 13-model registry).
The one well-posed gate, and it held.

**What ARI claims instead.** The "N× more sensitive" ratio is retracted; the results rest on
three well-posed claims:

1. **Instrument-class claim** — continuous vs threshold measurement.
2. **Structural vs contingent invariance** — see below, replacing the earlier
   discrete-attractor necessity claim.
3. **Triple universality `(s, b, κ)`** — the per-model fingerprint.

A slope ratio depends on the degeneracy of the baseline; a class claim backed by a
structural argument and a universal law does not.

**Discrete-attractor necessity — retracted, and not replaced by an equivalent.** This one
took two passes to get right, and both are recorded because the intermediate version was
circulated.

*First pass.* The original text read the 0.0000-vs-0.5 probe-purity result as proof that
only SEMQ can serve as an ARI probe. It does not support that. The 0.5 figure comes from
comparing *independently seeded* instances, and a competitor is free to freeze and publish
one codebook instead. Such a probe is deterministic, satisfies `encode(reconstruct(c)) = c`,
and at matched bitrate is at least as sensitive as SEMQ.

*Second pass — also wrong, and retracted here.* The correction claimed a VQ probe is
"technically viable but not verifiable," because near-ties in the codebook let two
algebraically identical distance formulas return different codes: 39% of symbols, and
`encode(reconstruct(c)) = c` failing outright. Those numbers are real but were measured on a
**constructed** codebook pairing centroids `1e-6` apart. Generalising them to shipped
codebooks was unsupported, and the "1.6% of assignments near a tie at a typical
configuration" figure supporting that step came from synthetic data at an atypical
configuration.

*What the measurement actually shows* ([`experiments/probe-verifiability/`](../experiments/probe-verifiability/RESULTS.md)):
on real k-means codebooks, `encode(reconstruct(c)) = c` **holds** at every configuration
swept, including one with 91% of assignments inside `1e-6` of a tie. Formulation
disagreement is real but small — 0% to ~5% of vectors, configuration-dependent.

**So SEMQ is not necessary for ARI.** What remains is a difference in how the guarantee is
obtained: SEMQ's is checkable from the operator's form in advance, a VQ's has to be
re-measured for each codebook shipped, and LSH is separately ruled out on sensitivity.
That is a preference, argued on verification cost rather than on impossibility. It is a
much weaker claim than either version it replaces. It has the advantage of being true.

**Standing caveat, not a retraction — the LSH comparison rests on one synthetic run.** The
~14× sensitivity gap that rules LSH out was measured on anisotropic synthetic vectors at
matched bitrate, not on a real corpus, and has not been replicated. It is the weakest number
still in service ([probe-validation](analysis/probe-validation.md) asserts it). Treat it as
indicative until it is re-derived the way the near-tie sweep was; if that re-derivation
refutes it, the LSH exclusion above moves from "ruled out" to retracted.

