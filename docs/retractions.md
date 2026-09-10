# Retractions

This ledger records withdrawn claims, the evidence against them, and the remaining supported interpretation.
It does not establish publication or peer-review status for the paper.

## Sensitivity relative to retrieval

Withdrawn claim: SEMQ is approximately 300 times more sensitive than retrieval.
The registered test required a SEMQ-to-recall slope ratio of at least 100.
Measured ratios ranged from 1.1–4.2 to infinity because Recall@10 could remain constant.
The ratio did not provide a stable comparison of detector sensitivity.

The fitted slope remained close to one. The empirical prefactor differed from the uniform-sphere prediction by 1.48–3.06 times.
The original five-percent prefactor criterion therefore failed.
The slope dispersion criterion, `std(b) ≤ 0.05`, passed with `std(b) = 0.028` across 13 models.
See [drift-sensitivity results](../experiments/drift-sensitivity/RESULTS.md).

Report the response curves and their assumptions. Do not restore the fixed sensitivity-ratio claim.

## Necessity of SEMQ

Withdrawn claim: only SEMQ can serve as an ARI probe.
The original comparison used independently seeded quantizers.
Their disagreement does not establish disagreement for a fixed, published codebook.
A frozen vector quantizer can satisfy `encode(reconstruct(c)) = c`.

A later correction also overclaimed that frozen vector quantizers were not verifiable.
Its 39-percent disagreement came from constructed centroid pairs separated by `1e-6`.
The related 1.6-percent near-tie estimate used synthetic data and an atypical configuration.
These measurements did not support a general claim about fitted codebooks.

In the real k-means sweep, the reconstruction property held in every tested configuration.
This included a configuration with 91-percent near-tie density.
Distance-formulation disagreement varied from zero to approximately five percent of vectors.
See [probe-verifiability results](../experiments/probe-verifiability/RESULTS.md).

SEMQ remains the specified embedding probe. The choice does not establish that alternative probes are impossible.
Verification cost and reproducibility must be assessed for the actual operator or published codebook.

## LSH evidence limit

The approximately 14-fold LSH sensitivity gap comes from one synthetic experiment at matched bitrate.
It has not been replicated on a real corpus.
Treat it as limited evidence, not a general exclusion of LSH.
See [probe validation](analysis/probe-validation.md).
