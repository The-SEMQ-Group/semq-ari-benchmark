# Retrieval outcomes study. Protocol v0.1-draft

**Status: draft. Not yet frozen.** No confirmatory episode has been collected.
The pilot in [PILOT.md](PILOT.md) runs first. The protocol is frozen as v1.0
after the pilot, before any confirmatory episode. Section 13 gives the freeze
procedure and the decisions that remain open.

Linear task: SEM-50. Draft written at repository commit `73139c8` (`origin/main`).

This study asks whether ARI diagnostics predict two operational events on a
fixed retrieval system: a loss of retrieval quality, and a benefit from
rebuilding the index. It reuses the frozen inputs and scorers of the
[regime-discrimination experiment](../regime-discrimination/RESULTS.md) and the
alarm conventions of the
[matched-budget detector protocol](../matched-budget-detectors/PROTOCOL.md).
A null result is a valid outcome.

---

## 0. Three questions, kept separate

Every episode is classified on three separate questions.

| Question | Measured by | Unit |
| --- | --- | --- |
| Q-REP: Did the representation change? | Code and vector scorers (section 2) | Frozen input, aggregated to episode |
| Q-RET: Did retrieval quality degrade? | Recall@10 and nDCG@10 against judgements (section 3) | Query, aggregated to episode |
| Q-FIX: Did rebuilding restore quality? | Recall@10 and nDCG@10 after each remedy (section 8) | Query, aggregated to episode |

An answer to one question does not answer another. Section 10 gives the wording
permitted for each.

---

## 1. Frozen inputs

### 1.1 Corpus, queries and relevance judgements

| Item | Value |
| --- | --- |
| Corpus | BEIR SciFact, `BeIR/scifact` config `corpus`, split `corpus`: 5,183 documents |
| Queries | `BeIR/scifact` config `queries`, filtered to queries with a positive test judgement: 300 queries |
| Judgements | `BeIR/scifact-qrels`, split `test`, score > 0 |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, output normalised |
| Reference configuration | fp32, CPU, 4 threads, batch 32 |
| Loader | `load_scifact` in [run_matrix.py](../regime-discrimination/run_matrix.py) |

Resolve and record the Hub revision of the encoder, the corpus and the qrels
with `ari.hub.hub_revision`. Write the cache manifest with
`write_cache_manifest` and key every cached array on `fingerprint()`. Refuse a
manifest mismatch.

### 1.2 Why SciFact

1. The reference embeddings, codes, retrieval metrics and their manifest exist.
   The reference Recall@10 is 0.7833 and nDCG@10 is 0.6451.
2. Six real serving conditions are already measured on it. Their table is in
   the paper appendix "Full SciFact condition table". New rows are comparable.
3. The corpus encodes on a CPU in about one minute. Section 9 and PILOT.md use this.
4. The judgements are real. A synthetic corpus pins Recall@10 at 1.0 and measures
   the degeneracy, not the instrument. See [retractions](../../docs/retractions.md).

The cost of SciFact is 300 queries. Section 3.2 gives the smallest loss this
query count detects. A larger BEIR set with a CPU-sized corpus is reserved as a
secondary set, and is not run unless the pilot stop rule in PILOT.md triggers
it. Candidates are SCIDOCS (1,000 queries, 25,657 documents) and FiQA-2018
(648 queries, 57,638 documents). The choice is open (section 13).

### 1.3 Conditions not in scope

The `time` axis (a 24-hour gap) and the `lib` axis are not in this study.
Hosted API encoders are not in this study. The encoder runs locally.

---

## 2. Probe, references and scorers

### 2.1 Probe

Primary probe: SEMQ QUANT, `n_bins = 8`, calibration percentile 0.99, calibrated
once on the reference document embeddings and frozen. This is the probe of the
SciFact table, produced by `semq_codes` in `run_matrix.py`.

Secondary probe: the canonical embedding probe, `n_bins = 2`, percentile 0.99,
from [ari-canonical-v0.1](../../spec/ari-canonical-v0.1.md). Results from the
two probes are reported in separate families and never pooled
(matched-budget protocol, section 0.3 and 4.1).

The calibration scale is never recomputed inside an episode. A recalibrated
comparison measures a different target.

### 2.2 Two reference objects

A diagnostic compares codes of a frozen probe input set against stored codes.
Two stored code sets are defined, and every scorer is computed against both.

| Reference | Stored codes | Question it answers |
| --- | --- | --- |
| `R0` | Codes of the probe inputs at study commissioning, reference configuration | Did the representation change from the original? |
| `R_idx` | Codes of the probe inputs as embedded when the current index was built | Does the query path agree with the index? |

Under an unchanged index `R0` and `R_idx` are identical. They differ after a
coordinated rebuild. The probe input set is the frozen corpus. Each episode
re-encodes the probe inputs through the episode's **query path** and compares
the result with `R0` and with `R_idx`. Corpus-side scoring alone cannot see a
query-only change (section 4, arm B).

Preregistered expectation: alarms against `R_idx` predict retrieval loss
better than alarms against `R0`. This is a hypothesis, not an assumption.

### 2.3 Scorers, by function name

All ARI quantities come from `ari/`. Baselines come from the matched-budget
scorer module. Every rate names its denominator.

**ARI family**

| Quantity | Function | Denominator |
| --- | --- | --- |
| HER, H̄ (bits), HER bootstrap interval | `ari.metrics.per_input`, `ari.metrics.aggregate` | inputs; bits |
| Code equality per input | `ari.code_metrics.code_diff` → `CodeDiff.codes_equal` | inputs |
| Coordinate change rate (symbol mismatch) | `CodeDiff.coordinate_change_rate` | coordinates |
| Bit Hamming distance and rate | `CodeDiff.bit_hamming_distance`, `CodeDiff.bit_hamming_rate` | code bits, padding excluded |
| Byte change rate, legacy | `CodeDiff.byte_change_rate` | bytes. Reported for continuity only |
| Changed coordinate indices | `CodeDiff.changed_coordinates` | per input |
| Symbols per coordinate | `ari.code_metrics.unpack_symbols`, `ari.code_metrics.bits_per_coordinate` | |
| Extent and location profile | `ari.change_profile.profile` → `ChangeProfile.group_change_counts` | groups of 64 coordinates |
| Displacement bounds | `ari.change_profile.profile(regions=...)` → `ChangeProfile.displacement` | per coordinate |
| Bound informativeness | `ari.bound_quality.summarize_bounds` → `BoundQuality` | `_of_all` and `_of_changed` |
| Threshold resolution | `ari.bound_quality.resolve_threshold` → `ThresholdResolution` | coordinates |
| Storage per unit | `ari.change_profile.reference_bytes`, `ari.code_metrics.bytes_for` | serialized bytes |
| Cost record | `ari.bound_quality.cost_record`, `ari.bound_quality.timed` | |

Displacement bounds require a SEMQ build with `Context.quant_regions`
(`ari.change_profile.supported`). Where absent, the profile omits them and
the report marks them unmeasured.

**Baselines**, from
[`experiments/matched-budget-detectors/scorers.py`](../matched-budget-detectors/scorers.py):
`sha256_rows` over `canonical_bytes`, `max_abs_diff`, `rel_l2`,
`coord_mismatch`, `cosine_distance`, `block_hash_mismatch`,
`float16_roundtrip_delta`, `random_projection_scorer`. Budgets: `budget_ari`,
`budget_sha256`, `budget_block_hash`, `budget_uniform_quant`,
`budget_float16`, `budget_projection`, `budget_raw_fp32`.

**Retrieval-side baselines**, from `run_matrix.py`: the stored top-10 list
comparison (`top10_identical` in `compare`). Its budget is 300 queries × 10
ids × 4 bytes = 12,000 bytes plus the stored query texts. A labelled-query
canary (Recall@10 on a fixed 100-query subset) is reported as a reference
detector only. It shares queries with the outcome and is not an independent
predictor.

### 2.4 Coordinate-order assumption

Every coordinate-wise scorer assumes that the reference and the current vector
use the same coordinate order and the same configuration. Location output
(`changed_coordinates`, `group_change_counts`) is meaningful only inside a fixed
basis. The rotation arms (section 4.4 and 4.5) violate this assumption on
purpose. Location results are reported for non-rotation arms only.

---

## 3. Outcomes and the practically meaningful loss

### 3.1 Outcome measures

Compute with `retrieval_metrics` in `run_matrix.py`, exact search by dot product,
`TOP_K = 10`:

- Recall@10 per query, and its mean.
- nDCG@10 per query, and its mean.
- Top-10 identical rate against the reference lists.

Each episode reports Δ = episode − reference, per query, with a paired
bootstrap over queries (`paired_bootstrap_ci`, 10,000 resamples, seed 0).

### 3.2 Threshold, chosen before any episode of this study

| Measure | Practically meaningful loss |
| --- | --- |
| Recall@10 | Δ ≤ −0.030 |
| nDCG@10 | Δ ≤ −0.020 |

Basis. The threshold is set at the smallest loss that the frozen 300-query set
detects with 80% power, given the per-query dispersion observed in the prior
experiment. The dispersion was read from the published int8 intervals in
`regime_matrix.json`, the widest real intervention measured. No episode of this
study was used.

Arithmetic (`n = 300`):

1. int8 Recall@10 interval [−0.0278, +0.0156]. Half-width 0.0217.
   sd = 0.0217 × √300 / 1.96 = 0.192.
2. Minimum detectable effect at 80% power, two-sided 5%:
   (1.96 + 0.84) × 0.192 / √300 = 0.031. At 90% power: 0.036.
3. int8 nDCG@10 interval [−0.0185, +0.0101]. Half-width 0.0143. sd = 0.127.
   MDE at 80% power = 0.021. At 90% power: 0.024.
4. Under a pure-loss model where a fraction *p* of queries loses its single
   relevant document, a 10,000-resample bootstrap interval excludes zero with
   probability 0.85 at *p* = 0.02 and 0.98 at *p* = 0.03 (Monte Carlo, 400 trials).
5. Under a churn model with 5% losses and 2% gains, net −0.03, power is 0.53.
   Churn widens the interval. The threshold is not a guarantee of detection.

Interpretation. On SciFact a loss of 0.030 in Recall@10 is nine of 300 queries
losing their only relevant document from the top 10, net of gains.

Labels for an episode:

- **degraded (point)**: ΔRecall@10 ≤ −0.030, or ΔnDCG@10 ≤ −0.020.
- **degraded (confirmed)**: degraded (point), and the 95% interval upper bound is below 0.
- **not degraded**: the 95% interval lower bound is above the threshold.
- **inconclusive**: neither.

The primary label is **degraded (confirmed)**. Failure to reject a difference
is not equivalence. An episode is called **not degraded** only through the
interval bound above.

---

## 4. Episodes and intervention arms

### 4.1 Episode

An episode is one complete run of a declared configuration over the frozen
input set, in its own process, with its own manifest and wall-clock stamp. This
is the definition of the matched-budget protocol, section 2. An episode
records:

1. The arm, the intervention parameters and any random seed.
2. The query-path embeddings of the probe inputs and of the 300 queries.
3. The document index used for retrieval, by fingerprint.
4. Scorer output against `R0` and `R_idx`.
5. Per-query Recall@10 and nDCG@10, exact search.
6. For rotation arms, the ANN result (section 4.4).
7. Package versions, thread count, dtype, host, and elapsed seconds per stage.

A replicate of a deterministic condition on the same host is a valid episode
for the false-alarm bound. It is not an independent trial for prediction
(section 7.1).

### 4.2 Arms

Arms are grouped by compatibility pattern. "Real" means a change a serving
system can undergo. "Synthetic" means a transformation applied to cached
embeddings. Synthetic episodes are labelled `synthetic: true` in every table.

| Arm | Pattern | Source | Index | Query path | Expected Q-REP against `R_idx` | Expected Q-RET |
| --- | --- | --- | --- | --- | --- | --- |
| A0 `proc` | unchanged | real | reference | reference, fresh process | no change | none |
| A1 `threads1`, `batch8`, `batch128` | unchanged nominal output | real | reference | changed execution | none on this host | none |
| B1 `bf16-q`, `int8-q` | query-only, stale index | real | reference | precision changed | change | unknown |
| B2 `swap-q` | query-only, stale index | real | reference | different 384-d encoder | change | loss |
| C1 `bf16`, `int8` | coordinated | real | rebuilt under condition | same condition | no change; `R0` changes | none measured before |
| C2 `swap` | coordinated | real | rebuilt with new encoder | same encoder | no change; `R0` changes | either direction |
| G1 `gpu_*` | as regime-discrimination | real | per condition | per condition | as measured | none measured before |
| S1 `noise-both` | coordinated | synthetic | noisy | noisy, same draw | change | dose-dependent |
| S2 `noise-q` | query-only | synthetic | reference | noisy | change | dose-dependent |
| S3 `rot-shared` | coordinated, invariant | synthetic | rotated by Q | rotated by Q | change against `R0`; none against `R_idx` | none, exact search |
| S4 `rot-doc`, `rot-q` | one-sided | synthetic | rotated (S4a) or reference (S4b) | reference (S4a) or rotated (S4b) | change | large loss |

Arms A0, A1, C1 and G1 reproduce the regime-discrimination conditions. G1 runs
only when a GPU host is available and is reported as a host change as well.

Alternative 384-dimension encoders for B2 and C2, with Hub revisions recorded:
`sentence-transformers/all-MiniLM-L12-v2`,
`sentence-transformers/paraphrase-MiniLM-L6-v2`,
`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`.

Noise for S1 and S2: isotropic Gaussian, standard deviation σ × mean norm, as
in [drift-sensitivity](../drift-sensitivity/README.md). Grid
σ ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}. Renormalise after adding noise. Each
episode draws its own seed.

### 4.3 Real serving interventions

Real interventions run the encoder under the declared setting in a fresh
process, with the `encode` function of `run_matrix.py`. A query-only arm (B)
encodes the 300 queries and the probe inputs under the setting, and retrieves
against the cached reference index. A coordinated arm (C) also re-encodes the
corpus and builds a new index; `R_idx` is then the codes of that index.

### 4.4 Shared orthogonal rotation, the invariance control

Procedure per episode:

1. Draw a 384 × 384 Gaussian matrix from the episode seed. Compute Q by QR
   decomposition in float64. Fix the sign of each diagonal element of R so Q
   is Haar-distributed.
2. Check ‖QᵀQ − I‖_max < 1e-12 in float64. Record the value.
3. Apply Q to documents and queries in float64. Cast both to float32.
4. Compute S = queries · documentsᵀ in float64 for the reference arrays and
   for the rotated arrays. Record max |ΔS|. Acceptance: < 1e-12.
5. Compute the same in float32. Record max |ΔS|. Acceptance: < 5e-5. Both
   operands are float32 sums of 384 terms, so rounding alone is of order 1e-5.
6. Run exact search on the rotated float32 arrays. Record the top-10 identical
   rate and ΔRecall@10. No acceptance threshold. Differences arise only at
   near-ties and are reported as a measurement.
7. Build an ANN index on the rotated documents and search with the rotated
   queries. Record the top-10 identical rate and ΔRecall@10 against the ANN
   result on the reference arrays.

ANN implementation: `faiss-cpu`, `IndexHNSWFlat(384, 32)`, inner product,
`efConstruction = 200`, `efSearch = 128`. Build and search seeds recorded.
The ANN result is compared with the exact result on the same arrays. A
difference between step 6 and step 7 is an implementation effect, not a change
in the metric. Both are reported. The dependency is an open decision
(section 13).

Under this arm every coordinate-basis scorer against `R0` reports a change.
Against `R_idx` it reports no change, because the probe inputs and the index
were rotated together. Retrieval is unchanged under exact search. This arm
therefore has a known answer to all three questions, and any scorer that
alarms against `R_idx` on it has a false alarm.

### 4.5 One-sided rotation, a compatibility intervention

Apply Q from section 4.4 to the documents only (S4a) or to the queries only
(S4b). Do not apply it to the other side. This models an index and a query
path in different bases. Retrieval is expected to collapse. The scorers against
`R_idx` are expected to report a change. This arm and S3 share the same Q per
seed, so the Q-REP reading against `R0` is identical and the Q-RET reading is
not. That pairing is the point of including both.

---

## 5. Alarm policy at the deployment-episode unit

The matched-budget protocol, section 2.1 and 3, defines the policy. It is
reused without change.

1. **Per-input alarm.** For code scorers: the code differs from the reference.
   For continuous scorers: the score exceeds a threshold.
2. **Deployment alarm.** The per-input alarm rate over the probe set exceeds a
   threshold calibrated on control episodes (arm A0).
3. **Threshold.** The smallest threshold whose achieved control alarm rate does
   not exceed the nominal rate. Primary nominal FPR 5%. Secondary 1% where the
   control count supports it.
4. **Degenerate null.** When controls give zero alarms, report the achieved
   FPR of zero with the one-sided upper bound `1 − 0.05^(1/N)`. Compare
   methods on sensitivity at zero achieved false alarms. Report ties as ties.
5. **Matched budgets.** Compare each scorer at the serialized bytes it retains
   per input, from the `budget_*` functions and `reference_bytes`. SHA-256 and
   float16 are the small and large anchors.
6. **Achieved alarm rates.** Report every comparison at the achieved alarm rate
   on the evaluation split, with its interval, never at a nominal rate the
   method cannot realise.

Both units, per input and per episode, are reported for every method.

---

## 6. Episode split

Three disjoint roles, as in the matched-budget protocol, section 2:

1. **Probe fitting.** The calibration scale, from the reference document
   embeddings. Fixed before any episode.
2. **Threshold calibration.** 60 control episodes and the calibration half of
   every intervention arm.
3. **Evaluation.** 300 control episodes and the evaluation half of every
   intervention arm.

Assignment of an intervention instance to a half is by the parity of
`sha256(arm || seed || parameters)`, computed before the episode runs. All
replicates of one instance go to the same half. Calibration and evaluation
share no seed, no rotation, no encoder swap and no control episode.

The 300 queries are used for outcomes in every episode. They are not split.
Query-level intervals describe query sampling only.

---

## 7. Analyses, fixed before collection

### 7.1 Episode-level prediction

The unit is the intervention instance. An instance is one distinct (arm,
parameters, seed). Replicates of a deterministic real condition form one
instance and enter the prediction analysis once. This prevents counting
identical outputs as independent trials.

For each scorer, each reference (`R0`, `R_idx`) and each probe:

1. Classify every evaluation instance as alarm or no alarm at the calibrated
   deployment threshold.
2. Cross with the outcome label of section 3.2. Report the 2 × 2 table:
   alarm × degraded (confirmed). Report inconclusive instances separately.
3. Report sensitivity to confirmed degradation and the alarm rate on
   not-degraded instances, each with a one-sided bound from its count.
4. Report the same table restricted to real instances and to synthetic
   instances. Never pool them into one headline number.

Real serving interventions are few. Their table is descriptive. No inferential
claim is made from fewer than 59 confirmed-degraded real instances. The
expected result on real interventions is zero confirmed degradations, matching
the prior experiment. In that case the predictive claim is untestable on real
interventions and the report says so (section 11).

### 7.2 Query-level paired analysis

For each episode, Δ per query against the reference, paired bootstrap over
queries, 10,000 resamples. Report the mean, the interval and the label. Use the
same query resamples for every scorer in one episode.

Dose-response for S1 and S2: ΔRecall@10 and the coordinate change rate against
σ, one curve per reference object.

### 7.3 Correlated coordinates

A coordinate change rate is one number per input. The 384 coordinates are not
384 tests and not 384 interventions. Location output is summarised by
`group_change_counts` and reported per arm. No per-coordinate inference is made.

### 7.4 Multiple comparisons

Family: scorers against ARI, within one probe and one reference object, on
sensitivity at the declared alarm policy. Holm–Bonferroni within the family.
Uncorrected values also reported. Comparisons across probes or across reference
objects are exploratory.

---

## 8. Rebuild evaluation and cost

Run on every evaluation instance labelled degraded (confirmed), and on a
matched sample of not-degraded instances of the same arms as a control for the
remedy procedure itself.

### 8.1 Remedies compared

| Remedy | Action | Applies to |
| --- | --- | --- |
| RB rebuild | Re-encode the corpus under the current query-path configuration. Rebuild the index. `R_idx` becomes the new codes. | all arms |
| RL rollback | Restore the query path to the index configuration. | arms where the index configuration is available |
| AL alignment | Fit an orthogonal map from the current query space to the index space by Procrustes on 500 frozen anchor documents. Apply it to queries. | all arms; expected to work on rotation arms only |
| NO nothing | Serve as is. | all arms |

Each remedy produces per-query Recall@10 and nDCG@10 on the same 300 queries.

### 8.2 Benefit

For each remedy: Δ(remedy − degraded) and Δ(remedy − reference), paired over
queries, 10,000 resamples.

- **helped**: the interval for Δ(remedy − degraded) has a lower bound above 0.
- **restored**: the interval for Δ(remedy − reference) lies entirely above
  −0.030 for Recall@10 and above −0.020 for nDCG@10.
- **harmed**: the interval for Δ(remedy − degraded) has an upper bound below 0.
- **inconclusive**: none of the above.

### 8.3 Cost

Record per remedy with `ari.bound_quality.cost_record` and
`ari.bound_quality.timed`:

1. Encode wall-clock seconds and documents encoded.
2. Index build seconds.
3. Storage written, bytes.
4. Host, thread count, dtype, package versions.

Report cost per remedy per arm as a range over instances. Report benefit with
its interval beside cost. A remedy with a smaller cost that also restores
quality is reported before the rebuild.

---

## 9. Sample size

### 9.1 Control episodes and the false-alarm bound

The one-sided 95% upper bound on the alarm rate after zero alarms in N episodes
is `1 − 0.05^(1/N)`.

| N | Bound |
| ---: | ---: |
| 12 | 22.1% |
| 59 | 4.95% |
| 60 | 4.87% |
| 150 | 1.98% |
| 299 | 0.997% |
| 300 | 0.994% |

Minimum N for a 5% claim: ln 0.05 / ln 0.95 = 58.4, so 59.
Minimum N for a 1% claim: ln 0.05 / ln 0.99 = 298.1, so 299.

Target: 60 calibration controls and 300 evaluation controls, 360 in total. Zero
evaluation alarms then support a sub-1% bound. Any alarm count is reported with
its wider interval.

### 9.2 Degraded instances and the miss bound

Zero misses in K confirmed-degraded evaluation instances bounds the miss rate
by the same formula. Target K ≥ 59 in the evaluation half for a 5% bound.
Degradation is an outcome, so K is not known in advance. The arms expected to
produce confirmed degradation are S4 and B2. Plan 120 S4 instances (60 `rot-doc`,
60 `rot-q`), so that about 60 land in evaluation. If fewer than 59 are
confirmed, report the achieved K and its bound.

### 9.3 Instances per arm

| Arm | Instances | Replicates per instance | Episodes | Encodes corpus |
| --- | ---: | ---: | ---: | --- |
| A0 controls | 360 | 1 | 360 | yes |
| A1 | 3 | 30 | 90 | yes |
| B1 | 2 | 30 | 60 | queries and probe inputs only |
| B2 | 3 | 5 | 15 | probe inputs and queries |
| C1 | 2 | 30 | 60 | yes |
| C2 | 3 | 5 | 15 | yes |
| S1 | 5 σ × 12 seeds = 60 | 1 | 60 | no |
| S2 | 5 σ × 12 seeds = 60 | 1 | 60 | no |
| S3 | 60 | 1 | 60 | no |
| S4 | 120 | 1 | 120 | no |
| G1 | 4 | 5 | 20 | yes, GPU host |

Total: 920 episodes. 600 run the encoder on a CPU host: 525 encode the
corpus and 75 encode the query path only. G1 runs on a GPU host.
PILOT.md converts this to wall-clock time once the encode time is measured.

### 9.4 Query-level power

Section 3.2 gives the arithmetic. The 300-query set detects a 0.031 loss in
Recall@10 at 80% power under int8-like dispersion. It does not detect a 0.01
loss. A real intervention with a 0.01 loss reads as inconclusive. This limit
is stated in every table that uses it.

---

## 10. Three conclusions and permitted wording

Each conclusion has its own evidence and its own permitted sentences. No other
wording is used in the report or in downstream documents.

### 10.1 Representation changed

Evidence: HER < 1 against `R0` under the frozen probe, with the control bound.

Permitted: "The representation changed under the frozen probe." State the
reference object and the probe. Not permitted: "degraded", "drifted harmfully",
"broke retrieval".

### 10.2 Retrieval degraded

Evidence: the label degraded (confirmed) from section 3.2, with the interval.

Permitted: "Recall@10 fell by X [interval] on the 300 judged queries. This
exceeds the preregistered loss of 0.030." Not permitted: any statement about
cause that the arm design does not fix, and any statement about the
representation.

### 10.3 Rebuilding helped

Evidence: the labels helped and restored from section 8.2, with cost.

Permitted: "Rebuilding restored Recall@10 to within 0.030 of the reference
[interval] at a cost of Y seconds and Z bytes." Permitted when another remedy
also restored: "Rollback also restored quality at a lower cost."

"Rebuilding was necessary" is permitted only when all of the following hold:

1. The instance is degraded (confirmed).
2. RB is labelled restored.
3. Every remedy in section 8.1 other than RB is labelled not restored, or is
   unavailable in that arm and the report says why.
4. The decision criterion in section 8.2 and the recorded cost are stated in
   the same sentence.

Otherwise write "rebuilding restored quality" and name the alternatives.

### 10.4 Prediction

Permitted, when section 7.1 supports it: "On the evaluation split, alarms
against `R_idx` at zero achieved false alarms had sensitivity S [bound] to
confirmed degradation on synthetic instances, and the real instances produced
no confirmed degradation." Not permitted: "early warning", "predicts harm",
or any claim about arms not run.

---

## 11. Reporting

The results document lists, in separate sections:

1. Beneficial cases: instances where the alarm preceded confirmed degradation
   and a remedy restored quality.
2. Harmful cases: alarms with no degradation and rebuilds that did not help, or
   misses of confirmed degradation.
3. Inconclusive cases, by count and arm.
4. One operational claim in the wording of section 10, or a documented null
   result stating which arms were run and what the bounds are.

Every table carries `synthetic`, the reference object, the probe, the split,
the instance count and the episode count. Real and synthetic rows are never
summed.

---

## 12. Exclusions, stopping and deviations

1. Exclude an episode only for a recorded collection fault: manifest mismatch,
   scorer error, failed encode, or a rotation that fails the checks in
   section 4.4 step 2 to 5. List every exclusion. Report analyses with and
   without exclusions.
2. Stop at the episode counts in section 9.3. Stop early only for a recorded
   collection fault. Do not inspect prediction performance during collection.
3. Record every deviation from this document in the results with its reason.
4. Pilot episodes are never pooled with confirmatory episodes.

---

## 13. Freeze procedure and open decisions

Freeze procedure:

1. Run PILOT.md to completion or to its stop rule.
2. Update sections 9.3 and PILOT.md timings from measured encode times. Do not
   change section 3.2 or section 10 from pilot outcomes.
3. Resolve the open decisions below in the document.
4. Set the status line to "frozen", record the commit, and bump to v1.0.
5. Collect confirmatory episodes.

Open decisions for the owner:

1. Primary probe: `n_bins = 8` (SciFact table) or `n_bins = 2` (canonical).
   This draft sets 8 as primary and reports 2 as secondary.
2. ANN dependency: `faiss-cpu` with the parameters in section 4.4, or
   `hnswlib`. This draft names `faiss-cpu`.
3. Secondary corpus: SCIDOCS or FiQA-2018, run only under the pilot stop rule.
4. Whether `R_idx` is adopted as the operational reference, or reported as
   exploratory beside `R0`.
5. Whether the labelled-query canary (section 2.3) stays in the report.
6. The three alternative encoders for B2 and C2, or a different set.
7. Whether G1 runs. It needs a GPU host and adds a host change.
