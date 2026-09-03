# ARI-Canonical-v0.1 — the canonical probe specification

**Status: FROZEN (v0.1-preview).** The probe, calibration rule, and reference model below
are fixed for v0.1. The per-model fingerprint registry ([`fingerprints-v0.1.csv`](fingerprints-v0.1.csv))
is published and **grows as sweeps run** — adding a model's `κ` does not change the frozen
probe.

For ARI reports to be third-party comparable, three artifacts must be publicly fixed:
the **probe quantizer**, the **input set**, and the **reference model**. This document
fixes the probe.

## 1. The probe quantizer `Q`

- **Family:** SEMQ **QBIN** (quantile binning) — a deterministic quantizer holding the
  discrete-attractor property `encode(reconstruct(c)) = c` bit-exact
  ([probe-validation](../docs/analysis/probe-validation.md);
  other frozen deterministic quantizers can hold it too — this one is canonical because
  its behavior follows from a declared rule plus one calibration scalar, which is what
  keeps verification cheap. Validated by the probe-purity experiment: independently
  seeded quantizers read ≈ 0.5 on the `same` control, this frozen probe reads 0.0).
- **Bins:** `n_bins = 2` (~2 bits per embedding dimension / 4 symbols after calibration).
- **Calibration:** scale `s` fixed at the **99th percentile** of a published reference
  distribution.
- **Determinism requirement:** `Q(v) = Q(v)` exactly, zero measurement variance. Any probe
  with non-zero cross-instance Hamming (LSH / PQ / OPQ all sit at 0.5) is **disqualified by
  construction.**

## 2. Design rationale — why QBIN n=2, and why not sweep operators

**Why QBIN and not the other SEMQ operators.** QBIN is the only SEMQ operator with a
closed-form drift law — `E[Hamming] = 4·dim·σ / (s·√2π)`, linear in σ — derived directly
from its calibrated bin-edge geometry. That derivation is what makes the `(s, b, κ)`
fingerprint *predictive*: you forecast drift at any σ without re-running the sweep. The
other SEMQ operators would yield empirical fits with no closed form behind them, losing the
property that makes the fingerprint useful.

**Why `n_bins = 2`.**

- **Analytical tractability.** n=2 is the setting where the closed form is exact: the
  distance from a component to its nearest bin edge is uniform on `[0, s/2]`, which is what
  produces the clean linear-in-σ law. More bins muddy the derivation.
- **Maximum sensitivity.** The bin edge sits at the calibration threshold, so the smallest
  perturbation that matters crosses an edge and flips a bit — the high-pass behaviour is
  strongest here.
- **It matches the physical failure mode.** The founding drift incident was a single-bit
  code flip (a vector moving just enough to flip one bit of its QBIN code and reorder the
  top-K). n=2 is the natural code for measuring single bin-edge crossings.
- **Standardisation simplicity.** One published scalar `s` per model, one fixed probe →
  maximal comparability across submissions.

**Why not sweep operators.** A public standard needs *one* fixed probe; multiple operators
would fragment the "one number per agent" comparison without improving the instrument, and
only QBIN carries the theory-backed predictive fingerprint. Characterising the other
operators (open item §6) is a *defensive completeness* task — useful only to answer "why
QBIN and not X?" in review — and its likely outcome (other operators fit differently, with
no closed form) reinforces keeping QBIN canonical. It is not on the critical path.

## 3. Published calibration scalars

The canonical `s` is the **99th-percentile scale calibrated on the reference distribution**,
which for v0.1 is the frozen input set **ARI-Bench-v0.1**. `s` is a model property:
**corpus-invariant to <3%** (measured across MS MARCO, FiQA-2018, and Natural Questions), so
the ARI-Bench value and the cross-corpus value agree within that tolerance.

The full registry of published `s` for all 13 panel models is
[`fingerprints-v0.1.csv`](fingerprints-v0.1.csv). Cross-corpus **invariance evidence** (a
subset, calibrated independently on three retrieval corpora):

| model | dim | s (99th pct, cross-corpus) | cross-corpus variation |
| --- | --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 1024 | 0.0779 | 0.4% |
| BAAI/bge-m3 | 1024 | 0.0820 | 1.6% |
| intfloat/multilingual-e5-large | 1024 | 0.0760 | 2.8% |
| sentence-transformers/all-mpnet-base-v2 | 768 | 0.0992 | 2.6% |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.1321 | 0.6% |

**ARI-Canonical reference:** `BAAI/bge-large-en-v1.5`, `s = 0.0775` on ARI-Bench-v0.1
(cross-corpus `s ≈ 0.0779`, within the <3% invariance).

## 4. The `(s, b, κ)` model fingerprint

The instrument response of a model is characterised by **three** scalars, all
model-properties independent of corpus (within retrieval domain):

| scalar | meaning | value / range | evidence |
| --- | --- | --- | --- |
| `s` | 99th-pct calibration scale | per-model (registry) | <3% across 3 corpora |
| `b` | linear-regime power-law slope | 0.979 ± 0.028 (≈ 1) | universal across 13 models |
| `κ` | angular-concentration multiplier | [1.48, 3.06] | [`../experiments/drift-sensitivity/`](../experiments/drift-sensitivity/) |

Prediction from the fingerprint: `E[Hamming](σ) ≈ κ · √(2·dim/π) · σ`. Publishing
`(s, b, κ)` alongside a model's retrieval-quality numbers (MTEB / MMTEB) lets any team
predict drift at any σ — including the BLAS-typical σ ≈ 1e-6 — **without re-running the
sweep.** Observed drift above the prediction is *attributable* to a downstream noise source.

**Measured fingerprints** (full `(s, b, κ)`, from the drift-sensitivity sweep):

| model | dim | s | b | κ |
| --- | --- | --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 1024 | 0.0775 | 0.994 | 2.887 |
| BAAI/bge-m3 | 1024 | 0.0825 | 0.994 | 2.774 |
| intfloat/multilingual-e5-large | 1024 | 0.0762 | 0.991 | 2.893 |
| openai/text-embedding-3-large | 3072 | 0.0515 | 0.932 | 1.652 |
| voyage/voyage-4-large | 1024 | 0.0804 | 0.979 | 2.489 |
| cohere/embed-v4.0 | 1536 | 0.0699 | 0.995 | 2.657 |
| mistral/mistral-embed | 1024 | 0.0812 | 0.915 | 1.484 |
| gemini/gemini-embedding-001 | 3072 | 0.0510 | 0.990 | 2.547 |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 0.0934 | 0.999 | 2.924 |
| mixedbread-ai/mxbai-embed-large-v1 | 1024 | 0.0783 | 0.977 | 2.546 |
| Snowflake/snowflake-arctic-embed-l | 1024 | 0.0823 | 1.007 | 3.058 |
| intfloat/e5-mistral-7b-instruct | 4096 | — | 0.960 | 2.065 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 3584 | — | 0.994 | 2.641 |

**13 models now carry a full `(b, κ)`** (11 of the 13 panel models plus the 2 sweep-only
decoders), spanning `κ ∈ [1.48, 3.06]` with `b` universal at 0.979 ± 0.028. **Two panel models,
`all-mpnet-base-v2` and `all-MiniLM-L6-v2`, are `floor_limited`**: their `κ` cannot be cleanly
fit (see §5). All still have a published `s`. Full machine-readable registry:
[`fingerprints-v0.1.csv`](fingerprints-v0.1.csv).

## 5. Calibration caveat (must ship with the spec)

The small-σ Hamming floor is a property of the **(model × calibration) pair**, not the
corpus — and it is **model-specific, not a dimension rule.** Two models, `all-mpnet-base-v2`
(768-d) and `all-MiniLM-L6-v2` (384-d), exhibit a **hard floor** (~1% of code bytes flip under
sub-ULP perturbation, σ-independent down to σ = 10⁻¹⁰): ~1% of their components sit exactly on a
quantizer rounding boundary. This is *not* a low-dimension effect — `nomic-embed-text-v1.5`
(768-d) has **no floor** and fits `κ` cleanly (0.999 slope). It is also not driven by
near-zero-component mass or value discreteness (both ruled out empirically). Because the floor
swamps the linear regime, these two models are `floor_limited` (no clean `κ`) — **but their ARI
is unaffected: both remain bit-reproducible (ARI = 1.000).** The floor lives only in the
synthetic drift-sensitivity sweep, not the reproducibility measurement. The canonical probe
therefore **must specify the calibration jointly with the reference model** — a bare `n_bins=2`
is underspecified.

## 6. Registry growth toward v1.0

The probe is frozen; these items **extend the fingerprint registry**, they do not change it.

- **`κ` for the 9 `s_only` panel models — done.** Measured via [`ari/tools/drift_sweep.py`](../ari/tools/drift_sweep.py):
  7 fit cleanly (voyage, cohere, mistral, gemini, nomic, mxbai, arctic); 2 (mpnet, MiniLM) are
  `floor_limited` (§5). 11 of the 13 panel models now carry a full `(s, b, κ)`.
- **Publish `s` for the 7B decoder embedders** (e5-mistral, gte-Qwen2) — their `κ`/`b` are
  measured but they were not run in the panel, so no ARI-Bench `s` yet. (OpenAI's `s` is now
  published — registry.)
- **(Optional, defensive.)** Run a one-off operator comparison characterising the other
  SEMQ operators, purely to document why QBIN n=2 is the canonical choice (see §2). Not
  required for v1.0 and not on the critical path — QBIN n=2 stays the sole canonical probe.
- Re-run the recall side of the drift-sensitivity contrast on a full dev-query set to
  tighten the instrument-class comparison.
