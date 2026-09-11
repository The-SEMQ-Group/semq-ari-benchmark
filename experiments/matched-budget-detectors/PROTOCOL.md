# Matched-budget detector study — frozen protocol v2

**Status: frozen 2026-09-11, before any confirmatory collection.**
Supersedes [PROTOCOL-v1.md](PROTOCOL-v1.md), which the pilot in `results/`
was scored under. Section numbering is unchanged so existing citations in
`run_pilot.py` still resolve; §2.1, §3.1 and §8 are new.

**Original v1 status line:**
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

**An episode** is one complete collection of the frozen input set under one
declared configuration, in its own process, recorded with its own manifest.
Two episodes differ in at least their process and wall-clock time. Documents
or prompts inside one episode are *not* episodes: they share the process,
the machine and the session, so they support corpus-level uncertainty and
nothing about run-to-run variation.

### 2.1 Deployment alarm policy, declared separately

The per-input unit above answers "did this embedding change". An operator
asks "did this deployment change", and the two give different answers from the
same data. Measured on the cached SciFact panel, comparing a raw-vector
SHA-256 against the canonical code under four benign conditions:

| | per input | per deployment |
| --- | --- | --- |
| raw SHA-256 false-alarm rate | ~0.45% | **50%** (fires on 2 of 4) |
| canonical code false-alarm rate | 0% | 0% |

Two orders of magnitude, same data, from a definition. So the deployment
policy is declared here rather than chosen after the results are in:

**A deployment alarm fires when the per-input alarm rate over the frozen input
set exceeds a threshold calibrated on control episodes.** It is not "any input
changed". A raw hash detecting a benign byte change is not an integrity
failure, and a policy that treats it as one measures the input set's size
rather than the deployment's state.

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

v1 quoted 1.83% at N=200 from a two-sided interval. For an upper bound on a
false-alarm rate the one-sided convention is the right one, and it is used
here. **200 control episodes do not support a 1% claim**; 299 do.

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

On the cached panel the canonical code gives **zero** alarms across every
benign condition, so its control distribution has no variance. This is the
expected case, not a surprise, and it has a consequence the study has to
accept in advance: **with a degenerate null there is no threshold to trade and
no FPR to fix.** A nominal "5% FPR" is unachievable, because no threshold
produces 5% false alarms.

Three rules follow, fixed here:

1. A method whose controls are alarm-free is reported at its **achieved** FPR
   of zero with the one-sided upper bound its episode count supports, never at
   a nominal rate it cannot realise.
2. Methods are then separated on **sensitivity at zero achieved false alarms**,
   which is a well-defined comparison and the one the data can support.
3. Ties among alarm-free methods are reported as ties. Noise is not injected
   into controls to manufacture a threshold, and a method is not credited for
   discrimination the null cannot demonstrate.

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

KL and JS take **logits**, and the functions are named `kl_from_logits` and
`js_from_logits` for it. The softmax inside them turns a row into a
distribution over a vocabulary, which is what logits are; an embedding
coordinate is not an unnormalised log-probability of anything, so softmaxing
one returns a number with no meaning. Nothing at runtime distinguishes the
two — both are float arrays — so no guard can catch the misuse and the name
carries the precondition instead. Support and zero handling are declared with
the implementation and covered by the oracle tests.

### 4.1 Every comparison stays inside one probe

The methods above are not all available at both probes, so the comparison is
run **per probe** and no result crosses between them:

| probe | methods compared |
| --- | --- |
| embedding (`n_bins=2`) | ARI family, vector distances, equality and block hashes, uniform scalar quantization, sketch, float16 anchor |
| logit (`n_bins=8`) | all of the above, plus `kl_from_logits`, `js_from_logits`, top-2 margin, token flip, top-k set change |

A table placing KL at the logit probe beside ARI at the embedding probe would
compare two different measurements of two different objects and report the
difference as if it were a property of the methods. §0.3 already forbids
quoting storage from one probe against detection from the other; this extends
the same rule to every comparison, and each probe carries its own family for
the correction in §8.

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

## 8. Analysis, fixed before collection

**Multiple comparisons.** The study compares roughly a dozen scorers across
two probes and several interventions. Families are **per probe** (§4.1),
since no comparison crosses probes. Within each probe the primary family is
the scorers against ARI at the declared alarm policy, corrected by
Holm–Bonferroni at the family level, with uncorrected p-values reported
alongside. Comparisons outside a primary family are labelled exploratory and
are not used to support a claim. The two probes' families are corrected
separately and their results are never pooled.

**Paired testing.** All scorers see identical episodes, so every comparison is
paired at the episode level and resampled by episode, never by input. An
interval over inputs within an episode measures corpus sampling and is
reported as such.

**Power and stopping.** Collection stops at the declared 300 control and 300
intervention episodes, or earlier only on a declared failure of the collection
itself. There is no interim analysis of detection performance and no stopping
on a result. If collection falls short, the achieved counts and the bounds
they support are reported, and the shortfall is stated rather than absorbed.

**Exclusions.** An episode is excluded only for a recorded collection fault —
a failed upload, a scorer error, a manifest mismatch — never for its scores.
Every exclusion is listed with its cause and the analysis is reported with and
without them.

**What a result may say.** With the null expected to be degenerate (§3.1), the
supportable comparative statement is about sensitivity at zero achieved false
alarms, at equal retained bytes, on declared interventions. It is not about
early warning, functional harm, or a need to rebuild an index.
