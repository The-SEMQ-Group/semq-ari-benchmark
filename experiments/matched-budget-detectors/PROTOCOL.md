# Matched-budget detector study — frozen protocol v2

**Status: frozen 2026-09-11, before any confirmatory collection.**
Supersedes v1, used for the pilot and retained at commit `57d91e7`:

    git show 57d91e7:experiments/matched-budget-detectors/PROTOCOL.md

Section numbering is unchanged for `run_pilot.py`; new sections are §2.1,
§3.1, §4.1 and §8. See the CHANGELOG for details.

**Original v1 status line:**
Repository commit at freeze: `a961b3516b58` (branch `research/ari-review-response`).

Evaluates ARI's detection, storage, runtime and verification tradeoffs against
simpler measurements. Negative results are valid outcomes.

---

## 0. Audit findings that constrain the design

These findings constrain the study's claims.

### 0.1 The existing cache is TinyLlama-1.1B on CPU

`experiments/decoding-reproducibility/results/cache/` holds six flat NPZ files
(`reference`, `proc`, `threads1`, `batched`, `bf16`, `int8`), each
`logits (539, 32000) float32` over 12 prompts.

The 32,000-token vocabulary and `decoding_matrix.json` identify this as
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` on CPU. The cache contains 539 steps
(11×48 + 1×11) over 12 prompts.

**Consequence.** Appendix I baselines are TinyLlama CPU results, not results for
the paper's reported decoders.

### 0.2 The cache is unreadable by its own loader, and unlabelled

`baselines.py::_cache()` expects `results/cache/<fingerprint>/`, but the files
are flat and lack a manifest. Their provenance was reconstructed from metadata.

**Consequence.** Use these arrays only for implementation tests and synthetic
diagnostics. Confirmatory collection must write a manifest.

### 0.3 There are two probes, not one

| | magnitude bins | bits/dim | where |
| --- | --- | --- | --- |
| embedding probe | 2 | 2 | paper §2 and Appendix A |
| logit probe | 8 | 4 | `baselines.py:75`, `run_matrix.py:63`, published Llama artifact |

The probes are specified and reported separately; storage and detection results
are never combined across probes.

### 0.4 The published ARI score compares packed bytes, not symbols

Measured directly against the SDK: `batch_encode` returns packed bytes at
exactly 2.00 bits/coordinate for `n_bins=2` and 4.00 for `n_bins=8`, at both
384 and 4096 dimensions. It never returns one entry per coordinate.

`stat_semq` computes `(codes_c != codes_r).mean(axis=1)`, which is therefore a
**byte** disagreement rate reported as symbol disagreement. Changing exactly
one coordinate of 4096 gives a byte rate of 0.000488 against a true coordinate
rate of 0.000244: **2x overstatement**, the coordinates-per-byte factor, which
would be 4x at 2 bits/dim. The effect saturates as changes become dense.

**Consequence.** The published ARI score is not coordinate-wise comparable.
Report these quantities separately:

- `ari_code_hamming` — bit Hamming distance over unpacked code bits
- `ari_symbol_mismatch` — fraction of coordinates whose decoded symbol changed
- `ari_byte_mismatch` — the published quantity, retained for continuity

### 0.5 The committed pilot cannot be checked against the current scoring

`results/pilot.json` and `results/pilot.scores.npz` were produced by an earlier
`_ari_scores` that unpacked symbols itself and calibrated with
`numpy.percentile`. Both now come from the SDK.

The two changes do not have the same consequence.

**Unpacking.** The hand-rolled reader took each byte's symbols high-order
first. It grouped the right bits into the right coordinates and only permuted
them within a byte, identically in both operands, so every aggregate rate it
produced was correct and every changed-coordinate index was wrong. This is
pinned by `tests/test_sdk_surface.py::test_symbol_order_within_a_byte_is_not_arbitrary`.

**Calibration.** `numpy.percentile` interpolates in float64;
`Context.calibrate` takes the percentile in float32. When the percentile falls
between two distinct float32 values the scales differ, which moves codes near a
bin boundary and therefore moves all three aggregate rates. Whether it does is
a property of the data, not of the change: on the cache in
`experiments/decoding-reproducibility/results/cache/` at the pilot's settings
the two scales are bit-identical (`9.41488265991211`) and every rate is
unchanged, while at dim 384, `n_bins=8`, percentile 0.999 they diverge.

**The pilot's own inputs are not in this repository.** `pilot.json` records its
cache as `/home/ubuntu/mbd/cache`, which is not the cache above and is not
committed, and the scale it recorded (`9.414882678985599`) is not what either
path produces on the cache that is here. Nothing in `pilot.json` can be
re-derived, so no claim that its numbers survive the change can be checked.

**Consequence.** Treat `pilot.json` as a record that a run happened, not as a
result. It is retained unregenerated, and no number in it is comparable with a
number produced by the current scoring path. Confirmatory collection starts
from freshly collected episodes with manifests.

---

## 1. Two questions, kept separate

**Q1 — exact representation change.** Any difference in a canonical vector is a
positive; identical arrays are serialization controls. SHA-256 is the exact
equality baseline.

**Q2 — condition-change monitoring.** Detect a declared serving intervention
against repeated-run variation under a fixed nominal configuration. This null
does not imply that baseline variation is harmless.

Neither question establishes functional harm. Positives are defined by the
declared intervention, not ARI disagreement; interventions with no measured
change are reported separately.

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

**Episode:** one complete collection of the frozen input set under one
configuration, run in its own process and recorded in its own manifest. Episodes
must differ in process and wall-clock time. Inputs within an episode are not
independent run-to-run trials.

### 2.1 Deployment alarm policy, declared separately

The per-input unit asks whether an embedding changed; the deployment unit asks
whether the aggregate rate exceeded a threshold. On the cached SciFact panel:

| | per input | per deployment |
| --- | --- | --- |
| raw SHA-256 false-alarm rate | ~0.45% | **50%** (fires on 2 of 4) |
| canonical code false-alarm rate | 0% | 0% |

**A deployment alarm fires when the per-input alarm rate over the frozen input
set exceeds a threshold calibrated on control episodes. A per-input byte change
is not itself a deployment failure.

Both units are reported for every method. Neither is presented alone.

## 3. Error rates and sizing

Primary nominal FPR **5%**. Secondary **1%**, reported only where the effective
sample size supports it.

Sizing is set by what zero alarms in N independent control **episodes** can
establish. The one-sided 95% upper bound is `1 - 0.05^(1/N)`:

| N control episodes | 0 alarms gives 95% upper bound |
| ---: | ---: |
| 12 | 22.1% |
| 200 | **1.49%** |
| 299 | **1.00%** |
| 2,995 | 0.10% |

v1 quoted a two-sided interval. This protocol uses the one-sided bound; 200
controls do not support a 1% claim, while 299 do when zero alarms occur.

**Collection target: 300 control episodes and 300 per intervention**, which
establishes a sub-1% upper bound only when zero control alarms are observed.
Any nonzero alarm count is reported with its corresponding wider interval.

The existing 12 prompts bound the rate only below 22.1%, which is why that
cache is pilot data and is never pooled with confirmatory episodes.

Thresholds on discrete scores are set at the conservative side of a tie: the
smallest threshold whose achieved control alarm rate does not exceed nominal.
Achieved held-out FPR is reported with an interval, never asserted as exactly
matched.

### 3.1 The null is expected to be degenerate

The canonical code gives **zero** alarms across every cached benign condition.
With this degenerate null, no threshold realizes a nominal nonzero FPR.

Therefore:

1. A method whose controls are alarm-free is reported at its **achieved** FPR
   of zero with the one-sided upper bound its episode count supports, never at
   a nominal rate it cannot realise.
2. Separate methods on **sensitivity at zero achieved false alarms**.
3. Report ties among alarm-free methods as ties; do not inject control noise.

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

**Budget-matched block hashes** — the vector split into B contiguous blocks,
each hashed and truncated so that B digests occupy the same bytes as the code
they are compared against. This is the baseline that answers "a hash, but
given the same storage and therefore able to localise too", and without it the
comparison hands ARI localisation for free.

The block-hash layout is serialized as two little-endian uint32 values
(`n_blocks`, `digest_bytes`) and therefore costs the declared 8 shared bytes;
the vector dimension, dtype, canonical byte format, and SHA-256 algorithm are
fixed by this protocol and are not per-example metadata.

**A float16 reference** — not budget-matched, and deliberately so. It is the
larger-storage accuracy anchor: what a monitor gets for 2 bytes per coordinate
instead of a fraction of one. It bounds how much the compressed methods give
up.

**Logit-only** — top-2 margin change, argmax token change, top-k set change,
and KL and JS computed in float64 log-space, validated against a
high-precision oracle on tiny perturbations, identical inputs and extreme
logits. These are not dismissed for float32 cancellation observed elsewhere.

KL and JS take logits; `kl_from_logits` and `js_from_logits` apply softmax
internally. Do not apply them to embeddings: the float-array type cannot be
checked at runtime, so the function name carries the precondition. Support and
zero handling are specified in the implementation and tested against the
oracle.

### 4.1 Every comparison stays inside one probe

Methods are compared **within each probe**; no result crosses probes:

| probe | methods compared |
| --- | --- |
| embedding (`n_bins=2`) | ARI family, vector distances, equality and block hashes, uniform scalar quantization, sketch, float16 anchor |
| logit (`n_bins=8`) | all of the above, plus `kl_from_logits`, `js_from_logits`, top-2 margin, token flip, top-k set change |

Each probe has its own comparison family for the correction in §8.

**Sketch** — seeded Gaussian random projection; its seed and projection
metadata count toward its storage budget.

Zero-norm vectors, ties, saturation, bin boundaries and padding are specified
per method in `scorers.py` and asserted in tests.

## 5. Storage accounting

Charge actual serialized bytes, rounding partial bytes up, including scale,
threshold, seed, codebook and manifest metadata. Report per-example and
amortized shared state separately; do not report theoretical packed size as
implemented size. Include a one-float32 top-2 margin as a storage baseline.

## 6. Verification

Score the *same frozen vectors* on local CPU and p5 CPU/GPU; this tests the
implementation, not cross-platform inference. Report bit-exact code/hash
agreement, floating-score error against declared tolerances, and alarm-decision
agreement. Include threshold-adjacent and quantization-boundary fixtures.
Unsupported combinations are marked unmeasured rather than inferred.

Report signature checking and independent recomputation separately, including
bytes transferred and dependency cost. Apply the same integrity and tolerance
options to simpler baselines.

## 7. Decision rule

Report an ARI advantage only when held-out results support it at comparable
achieved FPR and budget, with scope stated. Equal detection at lower cost,
inferiority and no separation are valid outcomes. Do not claim early warning,
capability damage or index-rebuild requirements without operational validation.

## 8. Analysis, fixed before collection

**Multiple comparisons.** Families are per probe (§4.1). Holm–Bonferroni is
applied within each primary family of scorers versus ARI at the declared alarm
policy; uncorrected p-values are also reported. Other comparisons are
exploratory, and probe families are never pooled.

**Paired testing.** All scorers see identical episodes. Test comparisons are
paired and resampled by episode, never by input. Input-level intervals describe
corpus sampling only.

**Power and stopping.** Stop at the declared 300 control and 300 intervention
episodes, or earlier only for a recorded collection failure. Do not inspect
detection performance interim. Report any shortfall and its resulting bounds.

**Exclusions.** Exclude episodes only for recorded collection faults (failed
upload, scorer error or manifest mismatch), never for their scores. List every
exclusion and report analyses with and without them.

**Permitted claim.** With the expected degenerate null (§3.1), compare
sensitivity at zero achieved false alarms and equal retained bytes on declared
interventions. Do not claim early warning, functional harm or index-rebuild
requirements.
