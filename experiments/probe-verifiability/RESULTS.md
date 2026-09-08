# Frozen vector-quantizer verification

The experiment compares two squared-distance formulas on fitted k-means codebooks.
It uses 5,000 SciFact abstracts encoded by `all-MiniLM-L6-v2`, with 384-dimensional normalized embeddings.
Half the embeddings fit the codebook; half evaluate it.
Each subspace has 256 centroids. The random seed is zero.
Data: [near_tie_sweep.json](results/near_tie_sweep.json).

## Results

A row is one probe vector. The reconstruction identity was checked under both distance formulas.

| subspaces | dims/subvector | near-tie <1e-6 | near-tie <1e-5 | median margin | forms disagree (symbols) | forms disagree (rows) | `encode(reconstruct(c)) = c` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 32 | 12 | 0.0450% | 0.3550% | 1.87e-03 | 0.0000% | 0.0000% | holds, both forms |
| 64 | 6 | 0.0700% | 0.7444% | 8.68e-04 | 0.0006% | 0.0400% | holds, both forms |
| 128 | 3 | 0.3044% | 3.0791% | 2.05e-04 | 0.0000% | 0.0000% | holds, both forms |
| 192 | 2 | 2.6206% | 13.7535% | 4.50e-05 | 0.0000% | 0.0000% | holds, both forms |
| 384 | 1 | 91.1873% | 99.5359% | 3.22e-07 | 0.0122% | 4.6400% | holds, both forms |

## Interpretation

Near-tie frequency increased as subvector dimension decreased.
The identity `encode(reconstruct(c)) = c` held in all tested configurations.
Formula disagreement remained configuration-dependent.
At 64 subspaces, 0.04 percent of rows disagreed. At 384 subspaces, approximately 4–5 percent disagreed.

Repeated k-means runs changed the last digits despite the configured seed.
A second run gave 4.36 percent row disagreement at 384 subspaces, compared with 4.64 percent in the table.
Do not interpret those digits as a stable population rate.

The earlier 39-percent symbol-disagreement claim used a constructed codebook with near-duplicate centroids.
It did not describe these fitted codebooks. The [retraction ledger](../../docs/retractions.md) records the correction.

## Limits

This is one encoder, corpus, and seed configuration.
Equivalent formulas are a proxy for arithmetic variation, not a direct cross-architecture measurement.
The one-dimensional configuration tests an extreme case.
These results support testing each codebook; they do not establish that frozen vector quantizers are unusable.

## Reproduce

From the repository root:

```bash
python -m pip install -e ".[selfhosted,data]" scikit-learn
python experiments/probe-verifiability/near_tie_sweep.py
```

The command downloads the model and dataset, then writes `experiments/probe-verifiability/results/near_tie_sweep.json`.
The SEMQ SDK is needed only if artifact signing is configured.
