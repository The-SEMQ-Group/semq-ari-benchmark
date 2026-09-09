# Matched-budget detector study — frozen protocol

**Status: frozen 2026-09-09, before any confirmatory evaluation.**
Repository commit at freeze: `a961b3516b58` (branch `research/ari-review-response`).

Addresses major concern 1 of the paper review: does ARI provide a useful
detection / storage / runtime / verification tradeoff against simpler
measurements? A negative result is an acceptable outcome and this document is
written so that it can be reported as one.

---

## 0. Audit findings that constrain the design

Established before writing anything below. Each changes what the study can
claim, so they are recorded first.

### 0.1 The existing cache is TinyLlama-1.1B on CPU

`experiments/decoding-reproducibility/results/cache/` holds six flat NPZ files
(`reference`, `proc`, `threads1`, `batched`, `bf16`, `int8`), each
`logits (539, 32000) float32` over 12 prompts.

Vocabulary 32,000 identifies the model by elimination: Llama-3.1-8B is 128,256,
Mistral-7B-v0.3 is 32,768, Qwen2.5 is about 152k. Only
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` matches, and `decoding_matrix.json` is the
sole `device: cpu` run, names TinyLlama, and reports the same 12 prompts. The
539 steps decompose as 11x48 + 1x11.

**Consequence.** The Appendix I baseline comparisons were computed on a 1.1B
CPU model, not on any decoder the paper reports. Nothing derived from this
cache may be presented as a property of the paper's decoders.

### 0.2 The cache is unreadable by its own loader, and unlabelled

`baselines.py::_cache()` delegates to `run_matrix.cache_dir()`, which resolves
`results/cache/<fingerprint>/`. The files on disk are flat, with no fingerprint
directory and no manifest, although `write_cache_manifest()` exists for exactly
that purpose. Provenance above was recovered by elimination, not read from the
artifacts.

**Consequence.** These arrays are usable for implementation testing and
synthetic diagnostics only. They are not admissible evidence for a
confirmatory result, and this study writes its own manifest for everything it
collects.

### 0.3 There are two probes, not one

| | magnitude bins | bits/dim | where |
| --- | --- | --- | --- |
| embedding probe | 2 | 2 | paper §2 and Appendix A |
| logit probe | 8 | 4 | `baselines.py:75`, `run_matrix.py:63`, published Llama artifact |

Different probes for embeddings and logits are defensible. Presenting them as
one declared probe is not. This study specifies them separately throughout and
never quotes a storage number from one against a detection number from the
other.

### 0.4 The published ARI score compares packed bytes, not symbols

Measured directly against the SDK: `batch_encode` returns packed bytes at
exactly 2.00 bits/coordinate for `n_bins=2` and 4.00 for `n_bins=8`, at both
384 and 4096 dimensions. It never returns one entry per coordinate.

`stat_semq` computes `(codes_c != codes_r).mean(axis=1)`, which is therefore a
**byte** disagreement rate reported as symbol disagreement. Changing exactly
one coordinate of 4096 gives a byte rate of 0.000488 against a true coordinate
rate of 0.000244: **2x overstatement**, the coordinates-per-byte factor, which
would be 4x at 2 bits/dim. The effect saturates as changes become dense.

**Consequence.** The published ARI score is not on the same footing as the
coordinate-wise baselines it is compared against. This study scores three
distinct quantities and never conflates them:

- `ari_code_hamming` — bit Hamming distance over unpacked code bits
- `ari_symbol_mismatch` — fraction of coordinates whose decoded symbol changed
- `ari_byte_mismatch` — the published quantity, retained for continuity

---

## 1. Two questions, kept separate

**Q1 — exact representation change.** Any difference in a declared canonical
vector is a positive. Identical arrays are a serialization control. Raw-vector
SHA-256 is a valid and strong baseline here, and is expected to be optimal by
construction: it detects every change and nothing else.

**Q2 — condition-change monitoring.** Detect a declared serving intervention
beyond the variation of repeated runs under a fixed nominal configuration.
Same-configuration repeats define the operational null even when their vectors
differ. Tolerating baseline variability is not a claim that such variability is
absent or harmless.

Neither target establishes functional harm. ARI disagreement is never used to
define ground-truth positives; positives are defined by the declared
intervention. Interventions that produce no measured vector change are reported
separately, alongside both results.

## 2. Alarm unit and splitting

**Primary alarm unit:** one embedding, or one complete prompt for decoders.
For prompts the frozen aggregation over teacher-forced token scores is the
**mean over steps**, declared here and not revisited. Token-level results are
reported as diagnostics only and never as independent trials.

**Splitting** is by input *and* collection episode. Calibration and test share
no prompt, prefix, reference or episode. Three disjoint roles:

1. probe fitting (scales, projections)
2. threshold calibration
3. final evaluation

Resampling is grouped by prompt and by episode. Reused reference pairs are not
independent and are resampled as blocks.

## 3. Error rates and sizing

Primary nominal FPR **5%**. Secondary **1%**, reported only where the effective
sample size supports it.

Prospective calculation over prompt-level controls, Clopper-Pearson:

| n controls | finest achievable FPR | 0 alarms gives 95% upper | TPR CI half-width at 0.90 |
| --- | --- | --- | --- |
| 12 (existing cache) | 0.0833 | 0.2646 | 0.170 |
| 50 | 0.0200 | 0.0711 | 0.083 |
| 100 | 0.0100 | 0.0362 | 0.059 |
| 200 | 0.0050 | 0.0183 | 0.042 |

**The existing 12 prompts cannot support a 5% claim**: their finest achievable
FPR is 8.3%, and a flawless control run bounds the FPR only below 26%. This is
why the cache is pilot data.

**Collection target: 200 control episodes and 200 per intervention at the
prompt level**, which supports the primary 5% with a TPR half-width near 0.04
and makes the secondary 1% reportable.

Thresholds on discrete scores are set at the conservative side of a tie: the
smallest threshold whose achieved control alarm rate does not exceed nominal.
Achieved held-out FPR is reported with an interval, never asserted as exactly
matched. If controls are bit-identical the null is degenerate; that is reported
as a limit on discrimination among methods rather than papered over with
injected noise.

## 4. Methods compared

All hyperparameters frozen before the test set is touched.

**ARI family** — `ari_code_hamming`, `ari_symbol_mismatch`, `ari_byte_mismatch`
(§0.4), each at its own probe (§0.3).

**Vector distances** — max absolute difference, relative L2, coordinate
mismatch fraction, cosine distance.

**Equality** — SHA-256 of canonical bytes, and SHA-256 after declared rounding
or uniform scalar quantization at bit budgets matched to ARI. Canonical bytes
fix dtype, little-endian byte order, C-contiguous shape, negative zero
normalized to positive zero, and NaN rejected rather than hashed.

**Logit-only** — top-2 margin change, argmax token change, top-k set change,
and KL and JS computed in float64 log-space, validated against a
high-precision oracle on tiny perturbations, identical inputs and extreme
logits. These are not dismissed for float32 cancellation observed elsewhere.

**Sketch** — seeded Gaussian random projection; its seed and projection
metadata count toward its storage budget.

Zero-norm vectors, ties, saturation, bin boundaries and padding are specified
per method in `scorers.py` and asserted in tests.

## 5. Storage accounting

Actual serialized bytes, partial bytes rounded up, plus scale, threshold, seed,
codebook and manifest metadata. Per-example and amortized shared state are
reported separately. Theoretical packed size is never reported as implemented
size. One float32 top-2 margin is included as a storage option, since storing
one scalar is a legitimate competitor to storing two logits.

## 6. Verification

The *same frozen vectors* are scored on local CPU and on the p5 CPU and GPU.
Re-running inference on another platform tests something else and is not
substituted for this. Reported: bit-exact agreement of codes and hashes,
floating score error against declared tolerances, and agreement of the alarm
decision. Fixtures include threshold-adjacent and quantization-boundary cases.
Unsupported combinations are marked unmeasured rather than inferred.

Signature checking and independent recomputation from raw vectors are reported
separately, with bytes transferred and dependency cost for each. Simpler
baselines get the same integrity mechanism and tolerance options as ARI.

## 7. Decision rule

An ARI advantage is reported only where held-out results support it at
comparable achieved FPR and comparable budget, with its scope stated. Equal
detection at lower cost is a useful result. Inferiority, or no clear
separation, is an equally valid outcome and is reported without hedging.

Nothing here is described as early warning or as a capability-damage detector.
That would need a separate operational validation which this study does not
perform.
