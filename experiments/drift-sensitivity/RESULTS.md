# Drift Sensitivity Benchmark — Results

Reference measurements from the 66-cell primary run (BEIR MS MARCO) and the 132-cell
corpus extension (NFCorpus + SciFact). Machine-readable data in [`results/`](results/).

## Fit units and status

The historical fits below, in [`results/*.csv`](results/), and in the
[fingerprint registry](../../spec/fingerprints-v0.1.csv) measure `Hamming` as the fraction of
**packed code bytes** that differ (`(codes != clean).mean()` on the SDK's packed output). At
`n_bins = 2` one byte holds four coordinates, so this rate is about four times the coordinate
change rate while changes are sparse, and less than four times as changes become dense
([`ari/code_metrics.py`](../../ari/code_metrics.py)). The paper's "From bits to coordinates"
paragraph distinguishes this byte convention from the coordinate change rate `rho`. The two are
not the same quantity.

- The historical byte-unit fits are preserved unchanged. They are the fits the registry publishes.
- A coordinate-unit refit exists per model in [`results/coordinate/`](results/coordinate/),
  produced by [`refit_coordinate.py`](refit_coordinate.py). Where a coordinate-unit fit exists it
  supersedes the byte-unit fit for that model for the response model and the inverse `sigma-hat`.
- Refit status: see [Coordinate-unit refit](#coordinate-unit-refit). The registry values
  themselves are frozen within v0.1 and are not rewritten.

## Pre-registered hypotheses and verdicts

The thresholds below were fixed **before any data existed**. We report the verdicts
honestly, then reframe — because two of the three gates turned out to be mis-designed for a
phenomenon that is *categorical*, not a slope ratio.

| # | Hypothesis | Pre-registered gate | Result | Verdict |
| --- | --- | --- | --- | --- |
| H1 | SEMQ is a far more sensitive σ-detector than retrieval | SEMQ Hamming slope ÷ Recall@10 slope ≥ 100× | min ratio 1.1 | **refuted (gate artifact)** |
| H2 | The Hamming curve matches first-principles theory | `(b, a)` within 5% of `(1, √(2·dim/π))` | slope matches; prefactor 1.65–2.89× | **refuted (prefactor only)** |
| H3 | The curve's form is model-architecture invariant | `std(b) ≤ 0.05` across models | std = 0.026 | **confirmed** |

### Why H1's gate is mis-designed (not a failure of SEMQ)

H1 is a *ratio of two slopes*. On a clean corpus, Recall@10 is a degenerate quantity — it
stays pinned at 1.0 across the whole fit window for some models, and barely moves for
others. So the ratio either explodes to infinity (recall slope = 0) or collapses toward 1
(recall wobbles). Neither measures SEMQ's actual sensitivity.

| model | Recall@10 slope | ratio |
| --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 18.65 | 4.2 |
| BAAI/bge-m3 | 51.63 | 1.4 |
| intfloat/multilingual-e5-large | 70.64 | 1.1 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 34.42 | 3.9 |
| intfloat/e5-mistral-7b-instruct | 0.00 | ∞ |
| openai/text-embedding-3-large | 0.00 | ∞ |

The robust phenomenon is **categorical**: recall is a threshold detector; SEMQ Hamming is a
continuous instrument. You cannot express that as a finite slope ratio (see Finding 1).

### Why H2's gate fails only on the prefactor (and that failure is a discovery)

The slope `b` matches theory (max error 0.068; 5 of 6 models within 4% of `b = 1`). The
gate fails **only** because the empirical prefactor `a` is 1.65–2.89× larger than
`a_theory = √(2·dim/π)`. That theory assumes embeddings are uniform on the unit sphere;
real embeddings concentrate. The multiplier **is** a measurable geometry property — κ (see
Finding 2). The gate refutes H2; the science discovers a new scalar.

## Finding 1 — SEMQ and retrieval recall are different classes of instrument

SEMQ Hamming grows monotonically and near-linearly with σ across five orders of magnitude.
Recall@10 stays flat, then cliffs near σ ≈ 3e-3. Example (`BAAI/bge-large-en-v1.5`):

| σ | SEMQ Hamming | Recall@10 |
| --- | --- | --- |
| 1e-7 | 9.4e-6 | 1.000 |
| 1e-6 (BLAS-typical) | 8.0e-5 | 1.000 |
| 1e-5 | 7.9e-4 | 1.000 |
| 1e-3 | 7.6e-2 | 0.981 |
| 1e-2 | 5.8e-1 | 0.819 |

At the BLAS-typical noise level (σ ≈ 1e-6, where production non-determinism actually
lives), SEMQ shows a clean signal an order of magnitude above its noise floor while
Recall@10 reads exactly 1.000. Full per-cell data: [`results/sensitivity_curves.csv`](results/sensitivity_curves.csv).

## Finding 2 — the power law holds; the prefactor encodes κ

`Hamming = a · σ^b` with `b ≈ 1` across all six models. The prefactor exceeds the
uniform-on-sphere prediction by a per-model **concentration scalar** `κ = a / √(2·dim/π)`.
Full table: [`results/encoder_fingerprints.csv`](results/encoder_fingerprints.csv).

| model | dim | slope b | a (empirical) | a_theory | κ |
| --- | --- | --- | --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 1024 | 0.994 | 73.7 | 25.5 | 2.89 |
| BAAI/bge-m3 | 1024 | 0.994 | 70.8 | 25.5 | 2.77 |
| intfloat/multilingual-e5-large | 1024 | 0.991 | 73.9 | 25.5 | 2.89 |
| intfloat/e5-mistral-7b-instruct | 4096 | 0.960 | 105.4 | 51.1 | 2.07 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 3584 | 0.994 | 126.1 | 47.8 | 2.64 |
| openai/text-embedding-3-large | 3072 | 0.932 | 73.1 | 44.2 | 1.65 |

κ lives in a narrow band **[1.65, 2.89]** (mean 2.49, std 0.49) across architectures that
share nothing. Higher κ means tighter angular clustering — more bin-edge crossings per unit
σ. The three BERT-family models cluster at κ ≈ 2.8 (shared backbone); the OpenAI API is the
low outlier (κ = 1.65, softer slope b = 0.93), consistent with stronger isotropy in its
post-training.

### κ is corpus-stable (corpus extension)

Repeating the sweep on NFCorpus and SciFact keeps each model's κ inside a tight band. Full
table: [`results/kappa_per_dataset.csv`](results/kappa_per_dataset.csv).

| model | κ range across 2 corpora | spread |
| --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 2.86 – 3.09 | 0.23 |
| BAAI/bge-m3 | 2.75 – 3.10 | 0.35 |
| intfloat/multilingual-e5-large | 2.79 – 3.04 | 0.25 |
| intfloat/e5-mistral-7b-instruct | 2.00 – 2.02 | **0.03** |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 2.56 – 2.60 | **0.04** |
| openai/text-embedding-3-large | 1.81 – 1.85 | **0.04** |

The 7B decoder embedders lock κ to within a few percent across corpora; the BERT family
shows modest 8–12% spread but stays inside its per-model band.

## Finding 3 — the form is model-architecture invariant (headline)

Mean slope `b = 0.978`, std `0.026` across six radically different architectures (0.024 on
the 132-cell extension). **The instrument response depends on the operator, not the
model.** This is what lets a single calibration protocol apply to any embedder.

## The fingerprint that ships

Three scalars characterise a model's drift response, all model-properties independent of
corpus:

| scalar | meaning | value / range (this 6-model sweep) |
| --- | --- | --- |
| `s` | 99th-percentile calibration scale | per-model (see spec) |
| `b` | linear-regime power-law slope | 0.978 ± 0.026 |
| `κ` | angular-concentration multiplier | [1.65, 2.89] |

> **Registry extension.** The values above are this document's original 6-model primary sweep.
> The published fingerprint registry has since been extended to **13 models** on the frozen
> ARI-Bench (see [`spec/fingerprints-v0.1.csv`](../../spec/fingerprints-v0.1.csv)): `b = 0.979 ±
> 0.028`, `κ ∈ [1.48, 3.06]`, with 11 of 13 panel models fully fingerprinted and 2 (mpnet, MiniLM)
> `floor_limited`. The slope stays universal; the κ band only widened as more models were added.

Publish `(s, b, κ)` once per model and any team can predict expected drift at any σ —
`E[Hamming](σ) ≈ κ · √(2·dim/π) · σ` — without re-running the sweep. Observed drift above
that prediction in a deployment is *attributable* to a downstream noise source (mixed
precision, batched routing, a library bump), giving ARI a quantitative null hypothesis.

## Coordinate-unit refit

[`refit_coordinate.py`](refit_coordinate.py) repeats the sweep on the frozen ARI-Bench-v0.1
inputs (1,000 texts, content hash `e9ec8b01…`) with the canonical probe calibrated once. It
records three rates per input per `sigma` from `ari.code_metrics.code_diff`: the coordinate
change rate `rho`, the bit Hamming rate, and the legacy byte rate. It fits
`log rho = log a + b log sigma` on the window `sigma in [1e-5, 1e-3]`. Intervals come from a
1,000-resample bootstrap over inputs. Residuals are reported per cell. The inverse `sigma-hat`
is checked only where the observed rate lies inside the fitted range. Per-input rates are kept
in an `.npz` beside each JSON. Seeds: noise 0, bootstrap 1.

Environment for the runs below: macOS arm64, Python 3.11.12, numpy 2.4.6, torch 2.14.0,
sentence-transformers 6.0.1, `semq` 1.5.1.dev32, one BLAS thread, CPU.

### Refit status per model

| model | registry status (byte units) | coordinate refit | result |
| --- | --- | --- | --- |
| sentence-transformers/all-MiniLM-L6-v2 | `floor_limited`, no `b`, no `κ` | done | floor-limited in coordinate units too; see below |
| BAAI/bge-large-en-v1.5 | `measured`, b = 0.994, κ = 2.887 | done | see below |
| all other registry models | `measured` or `kappa_only` | not run | byte-unit values stand |

### Measured unit ratio

Across the grid on both models the byte rate divided by the coordinate rate is 4.00 up to
`sigma = 1e-5`, then falls as changes become dense. The bit Hamming rate is 0.5 times the
coordinate rate to within 2e-5 at every cell. At `n_bins = 2` a changed coordinate almost always
flips one of its two bits; six inputs at `sigma = 1e-2` flip both bits of a coordinate. The paper's factor-of-four conversion holds in the sparse regime and does not hold at the
top of the grid.

| sigma | MiniLM byte/coord | bge-large byte/coord |
| --- | --- | --- |
| 1e-7 … 1e-5 | 4.00 | 4.00 |
| 1e-4 | 3.98 | 3.99 |
| 1e-3 | 3.91 | 3.89 |
| 3.16e-3 | 3.76 | 3.64 |
| 1e-2 | 3.34 | 2.97 |

### sentence-transformers/all-MiniLM-L6-v2 (dim 384, pilot)

The registry lists this model as `floor_limited`: about one percent of code bytes changed at any
`sigma`, so no `(b, κ)` was published. The refit measures the same floor in coordinate units.
[`floor_histogram.py`](floor_histogram.py) replays the floor draw and records which coordinates
make it up: [`sentence-transformers__all-MiniLM-L6-v2.floor.json`](results/coordinate/sentence-transformers__all-MiniLM-L6-v2.floor.json).

- Floor probe at `sigma = 1e-10`: coordinate change rate 2.68e-3 (0.27 percent), byte rate
  1.07e-2, bit rate 1.34e-3. About one coordinate per 384-d vector changes at any `sigma > 0`.
- Cause: every output vector has exactly two coordinates with `|x| < 1e-8`, 2,000 of 384,000
  (0.52 percent). All of them lie between 1.5e-36 and 1.4e-32; no coordinate is exactly zero.
  They sit on the probe's sign boundary, and under symmetric noise each changes sign with
  probability one half. Expected floor: 0.26 percent. Measured: 1,029 of the 2,000 changed
  symbol (51.5 percent), which is the 0.27 percent floor. The changed symbols are exactly the
  coordinates whose sign changed in the raw vector, and no coordinate above 1.4e-32 changed.
- The fraction with `|x| < 1e-6` is 0.78 percent. The further 0.26 percent of coordinates between
  1e-8 and 1e-6 did not change at `sigma = 1e-10`. The coordinate rate reaches 0.78 percent only
  near `sigma = 3e-4`, where the noise is large enough to cross them. The 0.78 percent is an
  upper bound on the sign-boundary set, not the floor.
- The re-encode of identical vectors is bit-identical, so this is not probe non-determinism.
- Raw coordinate fit on the window: `b = 0.287 [0.279, 0.295]`, `κ = 0.0056 [0.0052, 0.0061]`,
  `R² (log) = 0.86`. The window residuals run +0.21, −0.09, −0.22, −0.13, +0.23 in log units.
  These numbers describe a floor plus an onset, not a power law. They are not a fingerprint.
- Floor-subtracted diagnostic fit: `b = 0.501 [0.475, 0.530]`, `κ = 0.020 [0.016, 0.025]`,
  `R² = 0.92`. Still not linear inside the window; the response only leaves the floor above
  `sigma ≈ 1e-4`.
- Inverse `sigma-hat`: the observed rate range on the window is [4.0e-3, 1.5e-2]. Inside that
  range the inversion misses by up to a factor 7 (relative error 6.06 at `sigma = 3.16e-6`,
  whose rate falls inside the range because the curve is flat). Outside it the miss reaches a
  factor 286 at `sigma = 1e-2`. The inverse is not usable for this model.
- Verdict: `floor_limited` stands in coordinate units. The floor is a property of the encoder's
  output, not of the packed-byte measurement.

Historical byte-unit values for comparison: none. The registry has no `b` or `κ` for this model.

### BAAI/bge-large-en-v1.5 (dim 1024, reference model)

Run as a second model so a registry value could be compared directly. This is one run of one
model; it does not refit the registry.

Historical byte-unit values (registry, fitted on BEIR MS MARCO 3,000 passages): `b = 0.994`,
`κ = 2.887`. The refit is on the 1,000 frozen ARI-Bench inputs, so the byte-unit row below is
the like-for-like check of the corpus change; the coordinate row is the new quantity.

| fit unit | b | 95% CI | κ | 95% CI | a | R² (log) |
| --- | --- | --- | --- | --- | --- | --- |
| byte (registry, MS MARCO) | 0.994 | — | 2.887 | — | 73.7 | — |
| byte (refit, ARI-Bench) | 0.996 | [0.972, 1.024] | 2.933 | [2.434, 3.673] | 74.9 | 0.99997 |
| **coordinate (refit, ARI-Bench)** | **1.001** | **[0.978, 1.029]** | **0.779** | **[0.646, 0.973]** | 19.9 | 0.99998 |
| bit Hamming (refit) | 1.001 | [0.978, 1.029] | 0.389 | [0.323, 0.486] | 9.9 | 0.99998 |

- Window residuals (log units) for the coordinate fit: −0.008, +0.011, +0.003, −0.009, +0.002.
  The power law holds on the window.
- `κ_byte / κ_coord = 3.76`. The refit's `κ_byte / 4 = 2.933 / 4 = 0.733` and the registry's
  `2.887 / 4 = 0.722` both sit inside the coordinate interval. They are 6 to 7 percent below the
  coordinate point estimate, because the byte-to-coordinate ratio has already fallen to 3.89 at
  the top of the window.
- Floor probe at `sigma = 1e-10`: all three rates are exactly zero. `near_zero_coordinate_fraction`
  is 2.5e-5 (26 coordinates in 1,024,000). No floor.
- Inverse `sigma-hat` from the coordinate fit: inside the fitted rate range [1.9e-4, 2.0e-2]
  the relative error is at most 0.011 at every window cell. Below the window the inverse
  underestimates `sigma` by 13 to 50 percent (at `sigma = 1e-7` the observed rate is 9.8e-7,
  one coordinate in a million, and the curve bends). Above the window it is within 1 percent
  at `3.16e-3` and `1e-2` in coordinate units; the byte-unit inverse misses by 7 and 24 percent
  there, because the byte rate saturates first. These out-of-range cells are reported, not
  validated.
- Bootstrap standard errors: `b` 0.013, `κ` 0.081 (coordinate). The bootstrap resamples inputs,
  so the interval carries the between-input spread of the response and not the noise draw.

### Paper

The paper's "From bits to coordinates" paragraph in `docs/paper/latex/ari.tex` already records
the unit correction; per-model numbers come from `results/coordinate/<model>.json`.

## Caveats

- **Single probe.** Only QUANT `n_bins=2` is characterised here; other SEMQ operators may
  fit differently. Whether `b`'s universality holds across operators is open.
- **Thin query set.** After truncating the corpus, only a handful of qrels-matched dev
  queries remained; the Hamming side is robust (thousands of passages × 1,000 bootstrap),
  but the Recall@10 side would tighten with a full dev-query re-run. It does not change the
  qualitative shape of Finding 1.
- **κ is a global average.** It compresses per-cluster manifold geometry into one number.
