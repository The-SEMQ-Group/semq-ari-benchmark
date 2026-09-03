# Drift Sensitivity — data files

Reference measurements. See [`../RESULTS.md`](../RESULTS.md) for interpretation.

## `sensitivity_curves.csv`

Per-cell drift response — 66 rows (6 models × 11 σ) on BEIR MS MARCO.

| column | meaning |
| --- | --- |
| `model_id` | Hugging Face / provider model id |
| `dataset_id` | evaluation corpus |
| `sigma` | Gaussian perturbation magnitude |
| `hamming_mean` | mean fraction of SEMQ code bits flipped (clean vs noisy) |
| `hamming_ci_low` / `hamming_ci_high` | 95% CI on the Hamming mean |
| `recall_at_10_mean` | Recall@10 of noisy retrieval vs clean top-10 |
| `recall_at_10_ci_low` / `recall_at_10_ci_high` | 95% bootstrap CI (1,000 resamples) |

## `encoder_fingerprints.csv`

Per-model power-law fit (`Hamming = a · σ^b`) on the window σ ∈ [1e-5, 1e-3), 6 rows.

| column | meaning |
| --- | --- |
| `model_id` | model id |
| `dim` | embedding dimension |
| `slope_b` | fitted exponent (≈ 1) |
| `prefactor_a_empirical` | fitted prefactor |
| `prefactor_a_theory` | `√(2·dim/π)` (uniform-on-sphere null) |
| `kappa` | `a_empirical / a_theory` — the concentration scalar |

## `kappa_per_dataset.csv`

Per-(model, corpus) refit from the corpus extension — 12 rows (6 models × NFCorpus +
SciFact). Same columns as `encoder_fingerprints.csv` plus `dataset_id`. Shows κ is stable
per model across corpora.
