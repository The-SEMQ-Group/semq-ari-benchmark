# Probe choice and validation

The instrument-selection analysis and the two validation experiments whose
write-ups previously lived in the project whitepaper (since removed in favor
of the peer-reviewed paper, which compresses this material in its
appendices). Kept here because the spec and the experiment index cite it as
the record of *why this probe, and what its controls do and do not show*.
Related: [retractions.md](../retractions.md).

## Choosing the probe — what the class buys, and what it does not

A reproducibility probe has one hard requirement: **its reading must not depend on the
machine or the library that produced it.** ARI's whole purpose is to detect whether a
*subject* behaved identically across hardware. If the instrument also moves, its own noise
and the signal are inseparable.

Two earlier versions of this section turned that requirement into a claim that SEMQ is the
*only* probe that meets it. Both were wrong, and both are corrected here. [The retraction
ledger](../retractions.md) records the sequence; this section states what is left.

**The first argument** rested on the **discrete attractor property**,
`encode(reconstruct(c)) = c`, citing the probe-purity experiment as proof that only
SEMQ satisfies it. That experiment fits each baseline with a *different random seed*, so it measures
seed-stability. A competitor does not have to accept that constraint: freeze the codebook
once, publish it, and ship it. Measured at matched bitrate against such a probe, a frozen
product quantizer is bit-deterministic, satisfies `encode(reconstruct(c)) = c` exactly —
a centroid's nearest centroid is itself — and is *at least as sensitive* to perturbation as
SEMQ. None of determinism, idempotence, or sensitivity separates the two.

**The second argument** looked for the separation in floating-point reassociation, and
overstated how much of it there is. A vector quantizer assigns a code by `argmin` over
floating-point distances. Floating-point addition is not associative, so two algebraically
identical ways of computing the same distance can differ in the final bits. Where two
centroids are nearly equidistant from a point, that difference can flip the assignment — so
the probe's reading would depend on the library that produced it rather than on the subject
being measured.

That mechanism is real: on a constructed codebook that pairs
every centroid with a near-duplicate `1e-6` away
(`tests/integration/test_probe_arch_invariance.py` in the SDK), the direct form `(b − c)²`
and the expanded form `‖b‖² − 2b·c + ‖c‖²` disagree on 39% of symbols, and
`encode(reconstruct(c)) = c` fails under the expanded form — the form production libraries
use, because its cross term is a matrix multiply.

**But k-means does not produce that geometry, and on the codebooks it does produce, the
attractor property does not break.** Sweeping real k-means codebooks over 5,000 SciFact
abstracts, 256 centroids per subspace ([`experiments/probe-verifiability/`](../../experiments/probe-verifiability/RESULTS.md)):

| dims per subvector | near-ties within `1e-6` | median margin | forms disagree (rows) | `encode(reconstruct(c)) = c` |
| ---: | ---: | ---: | ---: | :--- |
| 12 *(standard)* | 0.045% | 1.87e-03 | 0.00% | holds, both forms |
| 6 | 0.070% | 8.68e-04 | 0.04% | holds, both forms |
| 3 | 0.304% | 2.05e-04 | 0.00% | holds, both forms |
| 2 | 2.62% | 4.50e-05 | 0.00% | holds, both forms |
| 1 *(degenerate)* | 91.2% | 3.22e-07 | 4.6% | holds, both forms |

Near-tie density is genuine and rises three orders of magnitude as the split gets finer.
It does not, by itself, break the probe: `reconstruct(c)` returns the centroid, whose
distance to itself is zero to within rounding, and that stays below the distance to any
distinct centroid even when the margin is tiny. Two centroids `1e-6` apart do not survive a
k-means fit — they compete for the same Voronoi mass and merge.

What does remain is small: the two formulations disagree on **0% to ~5% of vectors**
depending on configuration, roughly 1 in 2,500 at a plausible setting. Same input, same
published codebook, same machine, different reading — but rarely.

**SEMQ has no such race.** Its codes come from threshold comparisons against a calibrated
scale, not from a nearest-centroid contest. There is no pair of candidates to reorder, so
the assignment cannot depend on summation order, accumulator width, or lane layout.

**The claim, stated precisely.** SEMQ is *not* necessary for ARI, and this section no
longer argues that it is. A frozen VQ is a viable probe. What SEMQ offers is a guarantee of
a different kind, plus one baseline that is ruled out:

1. **Sign-projection LSH is categorically unsuitable.** At matched bitrate it is roughly
   14× less responsive to perturbation than SEMQ — an instrument that barely moves when the
   subject does. *(Measured on synthetic anisotropic data in one unreplicated run; the
   [retraction ledger](../retractions.md) records that caveat.)*
2. **A frozen VQ's invariance is contingent and must be audited per codebook.** It ranges
   from clean to ~5% of rows across configurations of the same algorithm on the same data,
   and nothing in a shipped codebook file announces which regime it is in.
3. **SEMQ's invariance is structural** — checkable from the operator's form before any
   codebook exists — and the SDK asserts it on Linux x86_64, Linux aarch64, and macOS
   arm64 on every commit.

The honest summary is that (2) is a *preference*, not an impossibility result. It is the
difference between a property you can verify in advance and one you have to re-measure for
every artifact you ship.

**Two practical properties compound this**, and they matter for a standard even though
neither is an impossibility proof:

- **The calibration artifact is a single scalar.** SEMQ's frozen state is one number, so a
  report can quote it inline and a verifier can check it by eye. A VQ probe must distribute,
  version, and pin a codebook of thousands of floats. That is a supply-chain artifact with
  everything that implies.
- **The tamper surface is correspondingly smaller.** A large codebook can be nudged to
  flatter a particular subject in ways that are hard to detect by inspection. A scalar
  cannot.



## Probe purity — seed-stability, and what it does and does not show

SEMQ re-encodes identically every time (Hamming 0.0000 exact). Independently seeded PQ,
OPQ and LSH instances sit at 0.5 normalised Hamming, the max-entropy point: two instances
of the same learned probe agree no better than chance.

**Read this as a statement about seeds, not about the probe class.** The experiment fits
each baseline with a different random seed, so it shows that a learned probe cannot be
reproduced from its recipe. It does not show that a *frozen, published* codebook fails —
that one is deterministic, and it also holds `encode(reconstruct(c)) = c` on every real
codebook we have measured (the probe-choice section above). What separates the classes is
not whether the guarantee
holds but how it is established: for SEMQ from the operator's form, for a VQ by auditing
each codebook shipped.

The result still matters operationally: it means a VQ probe can only ever be shipped as a
frozen artifact, never as a procedure, because re-running the recipe produces a different
instrument.

## Cross-process attribution — it catches what retrieval misses

On bit-deterministic hardware, SEMQ reports **zero drift with zero false positives**.
Under synthetic perturbation matching the x86 BLAS regime, SEMQ detects drift on the vast
majority of inputs at σ where FP32 cosine top-10 is essentially blind, and the measured
Hamming reproduces the drift-sensitivity predictions within a few percent. The founding
anecdote: a real deployed demo produced a different response hash on a cross-process replay
after a sleep/wake cycle, traced to multi-threaded BLAS without pinned threads — invisible
to retrieval, caught by SEMQ.

