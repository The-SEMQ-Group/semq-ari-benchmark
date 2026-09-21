# Drift Sensitivity — data files

Reference measurements. See [`../RESULTS.md`](../RESULTS.md) for interpretation.

## Units

The three CSV files below measure `Hamming` as the fraction of **packed code bytes** that
differ between clean and noisy codes. At `n_bins = 2` a byte holds four coordinates. The rate
is about four times the coordinate change rate while changes are sparse and less than that
when they are dense. The CSV values are preserved unchanged. The fingerprint registry
publishes these byte-unit fits.

Coordinate-unit refits live in [`coordinate/`](coordinate/), one JSON and one `.npz` per model.
Where a model has a coordinate-unit file, it supersedes the byte-unit fit for that model.
Status per model: [`../RESULTS.md#refit-status-per-model`](../RESULTS.md#refit-status-per-model).

## `sensitivity_curves.csv`

Per-cell drift response — 66 rows (6 models × 11 σ) on BEIR MS MARCO.

| column | meaning |
| --- | --- |
| `model_id` | Hugging Face / provider model id |
| `dataset_id` | evaluation corpus |
| `sigma` | Gaussian perturbation magnitude |
| `hamming_mean` | mean fraction of packed code **bytes** that differ (clean vs noisy); not a bit or coordinate rate |
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

## `coordinate/<model>.json` and `coordinate/<model>.npz`

Output of [`../refit_coordinate.py`](../refit_coordinate.py). The file name is the model id with
`/` replaced by `__`.

JSON fields:

| field | meaning |
| --- | --- |
| `unit_of_primary_fit` | `coordinate_change_rate` |
| `inputs` | input set name, count and content hash |
| `seeds` | `noise` (perturbation RNG) and `bootstrap` (resampling RNG) |
| `sigma_grid`, `fit_window` | the 11-point grid and `[1e-5, 1e-3]` |
| `s`, `dim`, `n_bins`, `calibration_percentile` | the probe as calibrated once on the clean embeddings |
| `sphere_prefactor` | `sqrt(2·dim/π)` |
| `near_zero_coordinate_fraction` | fraction of clean coordinates with `abs(x) < 1e-6`; an upper bound on the sign-boundary set, not the floor (see `<model>.floor.json`) |
| `floor_probe` | the three rates at `sigma = 1e-10`, below the grid |
| `cells` | per `sigma`: mean coordinate, bit and byte rates, and the byte and bit rates divided by the coordinate rate |
| `fits.<rate>` | for each of the three rates: `fit` (`b`, `a`, `kappa`, window residuals in log units, `r2_log`, dropped zero cells), `bootstrap` (95 percent percentile intervals and standard errors for `b` and `kappa`), `grid_residuals_log` over all 11 cells, and `inverse` |
| `fits.coordinate_change_rate_floor_subtracted` | the same fit after subtracting each input's floor-probe rate; a diagnostic, not a registry fit |
| `fits.<rate>.inverse` | `sigma-hat` per cell from the fitted curve, its relative error, a bootstrap interval, and `in_range`: whether the observed rate lies inside the rate range of the fit window |
| `environment` | library versions, platform and thread pins |

NPZ arrays: `sigmas` (11), `input_ids` (n), and for each of `coordinate_change_rate`,
`bit_hamming_rate`, `byte_change_rate` a `(11, n)` matrix of per-input rates plus a
`floor_<rate>` vector (n) at `sigma = 1e-10`. `noise_seed`, `floor_sigma` and `model_id` are
stored as scalars.

## `coordinate/<model>.floor.json`

Output of [`../floor_histogram.py`](../floor_histogram.py) for a floor-limited model. It replays
the sweep's noise draws so the floor cell is the same draw as in the `.json` above.

| field | meaning |
| --- | --- |
| `coordinate_change_rate`, `sign_change_rate` | fraction of coordinates whose symbol changed at `sigma = 1e-10`, and the fraction whose sign changed in the raw vector; `flipped_equals_sign_changed` says whether the two sets are identical |
| `below_threshold.<t>` | fraction of coordinates with `abs(x) < t`, and the fraction of those that flipped |
| `flipped_abs_x` | minimum, median, 99th percentile and maximum of `abs(x)` over the flipped coordinates |
| `per_input_below_1e-8`, `per_input_flipped` | per-vector counts, minimum and maximum |
