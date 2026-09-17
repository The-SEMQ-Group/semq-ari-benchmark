# Agentic Compounding of Representation Drift

**Hypothesis.** Per-call embedding drift that leaves single-step retrieval unchanged compounds over a multi-hop retrieval agent. H1: trajectory divergence grows with depth and reaches a macroscopic fraction. H2: recall@K (the standard reproducibility check) stays ~1.0 while the top-1 flips and SEMQ Hamming > 0 — the standard check is blind, SEMQ is not. H3: at sigma=0 the divergence is exactly 0, so a bit-reproducible embedder has no compounding.

## Summary

- **n_corpus**: 3633
- **n_hops**: 15
- **operating_sigma**: 0.0010
- **H1_depth_amplification_holds**: True
- **H2_standard_check_blind_holds**: True
- **H3_reproducible_zeroes_it_holds**: True
- **verdict**: H1 confirmed; H2 confirmed; H3 confirmed

## Hypothesis verdicts

| hypothesis | claim | key_stat | status |
| --- | --- | --- | --- |
| H1 | Trajectory divergence grows with depth (compounding) | divergence@T=0.115 (floor 0.1), flip/hop=0.013, x8.6 | confirmed |
| H2 | recall@K blind (~1.0) while top-1 flips + SEMQ sees drift | recall@K=0.9879, top1_flip=0.0133, hamming=0.07776 | confirmed |
| H3 | sigma=0 -> divergence 0 at every depth (repro zeroes it) | sigma0 max divergence=0.0 | confirmed |

## Divergence by depth (fraction of trajectories diverged)

| depth | sigma=0 | sigma=0.0001 | sigma=0.001 | sigma=0.003 |
| --- | --- | --- | --- | --- |
| 1 | 0.000 | 0.000 | 0.002 | 0.028 |
| 2 | 0.000 | 0.000 | 0.010 | 0.061 |
| 3 | 0.000 | 0.007 | 0.027 | 0.096 |
| 4 | 0.000 | 0.007 | 0.037 | 0.129 |
| 5 | 0.000 | 0.007 | 0.040 | 0.147 |
| 6 | 0.000 | 0.007 | 0.047 | 0.177 |
| 7 | 0.000 | 0.007 | 0.051 | 0.191 |
| 8 | 0.000 | 0.007 | 0.058 | 0.218 |
| 9 | 0.000 | 0.007 | 0.063 | 0.236 |
| 10 | 0.000 | 0.007 | 0.071 | 0.258 |
| 11 | 0.000 | 0.007 | 0.083 | 0.287 |
| 12 | 0.000 | 0.007 | 0.092 | 0.321 |
| 13 | 0.000 | 0.010 | 0.105 | 0.353 |
| 14 | 0.000 | 0.012 | 0.111 | 0.369 |
| 15 | 0.000 | 0.012 | 0.115 | 0.392 |

## Single-step drift (per sigma)

| sigma | recall@K | top1_flip_rate | SEMQ_hamming |
| --- | --- | --- | --- |
| 0 | 1.0000 | 0.0000 | 0.00000 |
| 0.0001 | 0.9986 | 0.0007 | 0.00813 |
| 0.001 | 0.9879 | 0.0133 | 0.07776 |
| 0.003 | 0.9661 | 0.0435 | 0.21941 |

## Notes

**H1 — depth amplification (headline).** Divergence = fraction of (seed, run) trajectories whose doc-id sequence differs from the deterministic (sigma=0) reference by a given depth. At the operating sigma it reaches 0.115 by depth 15, from a per-hop top-1 flip rate of 0.013 — an amplification of ~8.6x. A per-step drift that barely moves a single retrieval becomes a macroscopic divergence once it compounds over the trajectory.

**H2 — the standard check is blind; SEMQ is not.** At the operating sigma, recall@K of the top-K set stays 0.9879 — a practitioner running a retrieval-reproducibility check would declare the pipeline reproducible. Yet the top-1 result (which drives the chain) flips at rate 0.0133, and the SEMQ Hamming of the query is 0.07776 (> 0). The drift that predicts divergence is exactly what the standard set-overlap check cannot see.

**H3 — reproducible-by-construction zeroes it.** At sigma=0 the divergence is 0.0 at every depth: all compounding is attributable to the per-call representation drift. A bit-reproducible embedder (self-hosted proc=1.000, L2_02) therefore has zero agentic compounding — SEMQ diagnoses the risk, determinism removes it.

Dose-response: the divergence-by-depth table shows the effect scaling with sigma, from 0 (deterministic) up through the regime where recall@K is still ~1.0.
