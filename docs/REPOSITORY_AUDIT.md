# Repository audit — 2026-09-16

**Verdict: the core measurements are useful, but the repository is not yet a
fully reproducible scientific release.** This audit corrected implementation
errors, tied paper tables to saved evidence, and narrowed unsupported claims.
Passing tests and valid signatures do not establish the correctness of every
historical capture.

The review covers the existing working tree on `cleanup/public-artifacts`,
including the revised paper and evidence bundle. During the audit, the checkout
advanced to `b2cb8773bd1c1dbe3171c45abd07e9f23c953212`; its license notices,
infrastructure configuration, SDK checks, bound-quality support and new tests
were merged into the isolated audit copy before final verification. Existing edits were
preserved. Work was prepared and tested in an isolated copy. No GPU models,
paid APIs, cloud captures, or remote publications were run or changed.

## Findings and corrections

| Priority | Finding | Resolution and remaining work |
| --- | --- | --- |
| High | Self-hosted capture scripts called an immediate or end-of-session repeat `time`, despite the specification requiring a gap greater than 24 hours. | CPU output now calls it `post_burst_repeat`; SFR leaves `time` unmeasured. Paper and panel documentation no longer claim complete canonical temporal coverage. Historical signed reports are preserved; delayed captures are still required. |
| High | Archived ARI-E results predate estimator corrections, and the sampled trajectory export cannot reconstruct the full analysis. | Numerical contrasts and power claims are withheld from the paper. Historical result pages are labeled. Rerun against a pinned full dataset before republishing them. |
| High | The embedding table called the mean of marginal confidence-interval endpoints a 95% interval for ARI. | Removed the aggregate interval. A valid interval needs the joint per-input core measurements and a declared resampling design. All-one bootstrap intervals do not certify population determinism. |
| High | `aggregate_report` compared `same` with itself, forcing HER to one even for a changing API. | Requires a separate `baseline_codes` capture and checks its input count. The mock caller was updated. This is an intentional API change for callers of that function. |
| High | Incomplete harness runs could raise on non-overlapping run pairs; auxiliary metrics used different weights and case counts from outcome agreement. | All metrics now average within each case, then equally across cases. Missing controls remain unmeasured. Duplicate run keys and invalid outcomes are rejected; NaN intervals cannot report detection. |
| Medium | The abstract called clean-ranking overlap Recall@10 and implied causal amplification from two different decoding denominators. | Corrected to clean top-10 overlap, and distinguished teacher-forced token disagreement from free-generation disagreement. |
| Medium | Historical SciFact `cosine_mean` is a dot product and can exceed one. | Historical JSON remains unchanged; generated paper tables label it correctly. New captures report separate dot products and normalized cosine. Small-corpus top-k handling and input alignment checks were also fixed. |
| Medium | The paper described a fitted byte-change response as an exact coordinate-rate law; it called a discrete quantizer continuous and invoked a sensitivity Jacobian. | Declared the fitted approximation, its byte units, sparse-change conversion, and unpropagated uncertainty. Replaced the Jacobian claim with coordinate locality. Coordinate-rate refits remain necessary. |
| Medium | Several profile sizes, timings, movement percentages and comparative influence-spread claims lacked replayable artifacts here. | Removed those numerical claims. Retained the region-bound definitions and tested limitations. |
| Medium | SFR's script cast one loaded model between precisions, contrary to its documented fresh-load procedure; its input path was wrong. | Fixed repository-relative input resolution and separate subprocess loads for fp32 and bf16. No new GPU capture validates the revised procedure yet. |
| Medium | Metrics silently cast float inputs to bytes and could emit NaN from empty samples; report construction accepted inconsistent counts and dimensions. | Reject invalid types/shapes, empty measurements, invalid bootstrap parameters and inconsistent report inputs. Input sets reject truncation, duplicates and non-string data. |
| Medium | `run_report --submit` attempted a removed local scorer after potentially billed calls. | Fails before input loading or capture, with a pointer to `ari-leaderboard`. Reports remain unsigned until separately attested. |
| Medium | The wheel omitted the schema required by `python -m ari.run --validate`. | Packages the canonical schema as `ari.spec`; installed-wheel validation now works outside the source checkout. |
| Medium | The standalone verifier could crash on malformed input or a missing cryptography import. | Returns verification failure and tests report tampering, signer-key mismatch and malformed manifests without the SDK. |
| Medium | Bound summaries accepted invalid intervals and rounded different requested percentiles to the same key. | Validate aligned, ordered bounds and preserve distinct quantile labels; regression tests cover NaN, reversed bounds and shape mismatch. |
| Low | Documentation conflated code equality with float equality, attributed provider changes to caches/routing, and used unqualified universal/superiority claims. | Revised the principal panel, calibration, GPU and harness documentation. Historical decks are explicitly identified as stale presentation artifacts. |

## Evidence consistency

Six numerical LaTeX tables are generated by
[`build_tables.py`](paper/latex/build_tables.py). `--check` and
[`test_paper_evidence.py`](../tests/test_paper_evidence.py) fail when generated
output differs from the saved sources. All three figures now load their plotted
numbers from JSON, including the previously inline SciFact and prefix curves.

| Paper evidence | Local check | Limits |
| --- | --- | --- |
| Five embedding API rows and normalized table | Recomputed core means and normalized endpoints from fourteen report snapshots, pinned to leaderboard commit `4f49fe35d8f12fc26cde34d0cec4b6c00214798d`; verified every snapshot hash. | Summary reports, not raw vectors. Does not validate self-hosted `time` labels, API backend causes or calibration extrapolation. |
| SciFact table and figure | Generated from `regime_matrix.json`; checked all six available CPU code comparisons against stored HER. | CPU matrices are later re-encodings with recorded source hashes and SDK version. GPU codes and full vector caches are absent. Retrieval metrics were not regenerated from embeddings. |
| Iterative retrieval walk | Verified all six archive hashes; checked 15 hops, zero noiseless divergence, final divergence 0.115, and clean top-10 overlap 0.9879167. | Individual trajectories absent; config asks for 3,000 documents while analysis reports 3,633; historical source revision unknown. No seed-level interval can be recovered. |
| Hosted completion panel | Generated canonical rows by transcript name; recalculated complete-core means; figure reads saved prefix comparisons. | Raw transcript files named by the manifest are absent. Two endpoints lack `time`; the platform lacks `conc`. Cannot recompute rates or bootstrap intervals from raw completions here. |
| Internal logits | Generated per-model and range tables from three production-model JSON files; checked bf16 token and generation denominators and grouped control equality. | No new GPU inference; historical byte-rate fields are retained and labeled. “Early warning” is contemporaneous disagreement, not a validated predictor. |
| H100 batch study | Figure generated from saved batch summaries. | No new GPU execution or kernel profiling. Shared settings do not prove identical kernels. |
| Fingerprints | Recomputed 13 fitted slopes: mean 0.979, sample SD 0.027597; byte prefactors 1.484–3.058. | Raw coordinate-level calibration trials and fit uncertainty are absent. Noise-equivalent provider ordering remains exploratory. |
| Auxiliary baselines, rank profile, codebooks and power | Inspected saved summaries and qualified their scope in the paper. | Summary consistency is not a new reproduction. Matched-budget superiority remains unmeasured at confirmatory scale. |
| Ten signed experiment reports | Standalone signature, report digest and named local input checks pass. | Keys were taken from the artifacts, not independently authenticated. Some manifests bind summaries or incomplete input sets. Signatures prove integrity under a key, not scientific validity. |

The older [advisor review](paper/REVIEW.md) remains as a dated record. Its
“abstract unchanged,” unavailable-test and inline-figure notes are superseded.
The paper retains the newer historical `v0.1-preview` citation and distinguishes
it from the subsequently revised working draft. Assign a release only after committing the final code,
paper, generated tables and evidence bundle.

## Mathematical checks

- Hamming uses the popcount of XOR, not a count of unequal packed bytes.
  Existing SDK tests cover symbol unpacking, padding, chunk boundaries and
  coordinate movement bounds; new public tests check report construction.
- For binary independent outcomes on one case, mean self-agreement minus
  cross-agreement equals `(p_A - p_B)^2` in expectation. A regression enumerates
  all two-repeat outcomes and verifies that identity. The sample estimator can
  be negative and does not attribute a causal mechanism.
- Case bootstraps condition on observed runs and assume cases are suitable
  independent sampling units. Unequal or missing repeats require explicit
  weighting; outcome-dependent missingness can still bias the result.
- Two changed bits per coordinate imply `H/(2d) <= rho <= min(H/d, 1)` for the
  canonical probe. Inverting historical byte fits through this bound adds an
  approximation and does not supply a statistical confidence interval.
- A nonsignificant Recall@10 difference is not equivalence. An all-one finite
  sample is not a proof of determinism. Quantizer-boundary changes do not
  establish semantic change or functional harm.

## Validation

Test environment: macOS arm64, Python 3.14.6, NumPy 2.5.3, pytest 9.1.1.
SEMQ Python bindings came from local SDK commit
`80e4b2c5adfb4ea874b81a7414739444caabd232`, with a clean Release native build
and sanitizers disabled. This does not validate every supported Python/platform
combination or a published SDK wheel.

- Full suite with SDK: **189 passed**.
- Public-only suite: **100 passed, 10 skipped**; skips are SDK-dependent modules
  and signer paths. Standalone verifier and paper evidence tests run publicly.
- Mock pipeline on all 1,000 frozen inputs: report schema validation passes;
  the frozen content hash matches the expected input set.
- Wheel built and installed into a separate temporary directory: the mock
  pipeline and schema validation pass outside the source tree.
- Six paper tables pass the generated-output check. Available CPU code hashes,
  dimensions and HER values pass their evidence checks.
- All three figures regenerated. Two-pass LaTeX compilation succeeds, with the
  main body ending on page 4. The local style is an adapted 2025 template;
  current venue compliance is not established.

## Work required before stronger claims

1. Capture self-hosted `time` after a recorded gap greater than 24 hours,
   preserving baseline codes, calibration, inputs, environment and timestamps.
   Publish corrected reports with explicit supersession; do not overwrite old
   signed artifacts without a new attestation.
2. Pin and retain the complete ARI-E outcome dataset used by the current
   estimator. Recompute contrasts and power, including false-positive coverage
   under small case counts and unequal repeats.
3. Fit the noise response against actual coordinate-change rates, report fit
   uncertainty and validate the range used for inversion. Until then, retain
   the exploratory qualification on normalized rankings.
4. Retain hosted raw transcripts and GPU vectors/codes, or provide a durable
   authenticated archive and replay procedure. Resolve the retrieval-walk
   corpus-size discrepancy and missing source revision.
5. Review historical slides and secondary prose before reusing them. Create an
   immutable release and verify the current venue template before submission.

Primary references checked include [LLM-42](https://arxiv.org/abs/2601.17768),
[structured GPU residuals](https://arxiv.org/abs/2511.00025),
[numerical nondeterminism](https://arxiv.org/abs/2506.09501), and
[Open-SWE-Traces](https://arxiv.org/abs/2606.16038). These support the cited
context, not the correctness of this repository's individual captures.
