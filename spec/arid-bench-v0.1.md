# ARI-D-Bench v0.1 — decoding-layer reproducibility for hosted LLM APIs

**Status:** adopted, implementing · proposed August 2026, adopted 2026-08-31 ·
rollout items 1–2 complete (power sim, frozen prompt set); item 3 (dry run)
measured except the rehosted `time` cell; item 4 (fp32 references) and the
first panel pending. Graduates ARI-D from a research protocol to a leaderboard
measurement.

**Summary.** The Decoding tab exists and is empty, while the decoding layer is the
one the community actually asks about: *is my provider silently serving me a
degraded model?* The research half is done — five models measured, the metric
lessons learned, the amplification mechanism confirmed. What is missing is the
productized protocol: a frozen prompt set, a condition set that works against a
black-box API, detectors that need no logits, and rows that pass the same scorer,
pin and attestation gate every Representation row already passes. This document
freezes that protocol. It changes no code by itself.

---

## 1. Why this layer, why now

Serving stacks change under their customers without notice: precision tiers,
quantized variants, batching kernels, backend swaps. The community has a name
for the symptom — "the model got worse overnight" — and no way to substantiate
it; providers have a denial, and no way to substantiate that either. The
phenomenon deserves its own name: **silent serving drift**.

The industry has started shipping control surfaces for it. Routing products now
expose a `quantizations` field so a caller can restrict which precision tier
serves their traffic. Nobody has shipped the instrument that confirms the
control works — a control surface without a measurement behind it is a
checkbox. This bench is the instrument, and it takes no side: it does not argue
that any provider drifts, it measures which ones do, on pinned inputs, with
signed receipts a third party re-checks without trusting us.

The decoding experiments give the mechanism a number: under a bf16 serving
change, fewer than 1% of individual token decisions flip, but 21–35% of whole
generations end up different, because one flipped token derails everything after
it. Small infrastructure drift is not a small problem at the generation level —
which is precisely the level a hosted API exposes. That amplification is what
makes a black-box protocol workable at all.

## 2. What changes against the research protocol

The research instrument reads the full logit vector. A hosted API does not
expose logits; some expose top-k logprobs (OpenAI, top 20), some expose nothing
of the kind (Anthropic). Two consequences:

1. **The protocol is detector-diverse by necessity.** Token sequences are
   observable for every API, so the primary detectors are sequence-level.
   Logprob detectors run where the surface exists and read `untested` where it
   does not — the same honest cell the Matrix already renders.
2. **SEMQ does not appear in the measurement — by decision, not only by
   impossibility.** The findings already settled this layer: the baselines work
   concluded "to detect a precision change, ship the margin", the
   most-sensitive-probe claim is withdrawn, and the harness layer set the
   precedent that where the measured object is not a vector, forcing a
   quantizer in is decoration. A bench that measures with plain, universally
   checkable statistics also cannot be accused of being rigged toward its
   operator's product — the instrument's neutrality is the verdict's
   credibility. What SEMQ contributes here is the project's surviving thesis:
   the **verification frame** — frozen inputs, content-hash pins, KMS-signed
   reports, independent re-checking with no stack of ours installed.

3. **The bench measures the layer at the API boundary, and says so.** The
   decomposition's case for ARI-D as a distinct measurement rests on the
   instrument reading signal where token output is identical — and that claim
   stands untouched: it is about the white-box instrument, which keeps its own
   territory (self-hosted, logits, and the `max_dim` engineering prerequisite
   that belongs to that method). None of that signal crosses an API boundary,
   so this bench measures what does cross it: the drift a customer can
   actually observe. The bench and the instrument produce **different
   quantities under the decoding layer's name**, and a bench row and an
   instrument reading are never shown as the same column.

One research lesson binds the metric design: exact-match rates saturate under
real drift, while graded statistics separate degraded regimes. That is a
division of labor, not a contradiction with the exact headline. The customer's
question — *does it give me the same answer twice?* — is binary by nature, and
the headline answers exactly that. Saturation is a problem when the job is
telling degraded regimes *apart*, which is what the graded detectors are for:
diagnostics reported beside the headline, never averaged into it.

## 3. Subjects

A row is a **(model, provider) pair**, in two classes:

- **`vendor`** — a closed model served by its own vendor (gpt-*, claude-*,
  gemini-*). Only self-consistency is measurable.
- **`rehosted`** — an open-weight model served by a third party (Llama, Qwen,
  DeepSeek on Together, Groq, Fireworks, DeepInfra…). Here a **reference** is
  also measurable: we generate the same completions from the published weights
  in fp32 under our control, and compare. This class answers the quantization
  question directly, and the same model across several providers turns serving
  differences into a column readers can scan.

## 4. The frozen prompt set

`ARI-D-Bench-v0.1`: **100 prompts**, committed as JSONL
(`data/arid-bench-v0.1.jsonl`), pinned exactly as ARI-Bench is — SHA-256 over
the ordered `input_id\0text\n` records, every prefix hash valid so a reduced run
can prove it measured the first N (`prefix:N`, marked not comparable).

**FROZEN** (rollout item 2):
`content_hash = af5b1a2bb35b9fa5f8d4e8f05b85a98f495a3edeb93e95da9c9ced6d783e0d6f`.
See `data/README.md` for the verify command and provenance.

Composition, 20 each: factual QA, code generation, multi-step reasoning,
instruction-following over a supplied passage, open-ended prose. The mix is not
sacred; what is sacred is that it freezes. Prompts are single-turn, no system
prompt, English, 20–200 tokens long. Enriching the set toward decision-dense
prompts (where top-2 margins run small and drift is most visible) is a v0.2
research question, not a v0.1 blocker.

**The prompts are authored for this bench, not sampled from a public dataset.**
That is a deliberate choice, for three reasons. First, the bench measures a
provider against itself, not capability — the prompt is a stimulus, and any
compositional bias applies to every row equally and cancels in the Δ-vs-same
comparison. Second, prompts from well-known benchmarks are exactly the strings
most likely to collide with other users' traffic and provider-side prompt
caches — a direct confounder for a byte-reproducibility measurement; authored
prompts, unpublished before the freeze, start cache-cold by construction.
Third, authored text carries no upstream licenses and no PII, which real-user
prompt corpora do. The cherry-picking objection is answered structurally, not
reputationally: the set froze (hash above) before any provider was measured,
and every row runs the identical set. What authorship does *not* guarantee is
that the set discriminates — a too-easy set saturates `exact_generation` at
1.000 everywhere and separates nothing. The dry run checks this (§13, item 3)
before the panel spends money on it.

## 5. Generation protocol

Per prompt, per condition: `temperature=0`, `top_p=1`, `max_tokens=128`, no stop
sequences, no tools, no system prompt. Where the API accepts a seed, it is set
and recorded. Every response is captured whole — token sequence, finish reason,
any logprobs, and the provider's version metadata (`system_fingerprint`, model
build headers) — because attributing a `time` change to a disclosed backend swap
versus a silent one is exactly the kind of claim the row exists to support.

**Transcripts are retained and bound.** The float-detector gate exists because
raw embeddings were never kept, so three detectors can never run retroactively.
Decoding transcripts are small text; the manifest binds them by digest alongside
the prompt set. Any detector defined later runs against the same evidence.

## 6. Conditions

The comparable core mirrors ARI-R so the two layers read the same way:

| condition | meaning at this layer | repeats |
| --- | --- | --- |
| `same` | k back-to-back calls, one session — the provider's own floor | k=8 |
| `proc` | fresh connection/session | k=8 |
| `conc` | measured calls dispatched at a sustained 64 requests in flight | k=12 |
| `time` | repeats across a >24 h gap | k=8 |

Repeats are set high on purpose. The power analysis for the harness layer
showed repeats sharpen the control more than adding cases does, and here the
case count is already frozen at 100 — so the marginal dollar goes to k. The
simulation in §13 validates these numbers before anything is spent.

The `conc` burst size is 64 because that is the burst the evidence comes from:
the representation layer's concurrency effect (an API collapsing from 0.6 to
0.2) was measured under bursts of 64, and a dozen concurrent calls from one
client may not move a hyperscale batch at all. The 64 needs no filler traffic:
the condition's 1,200 measured calls (100 prompts × k=12) are simply dispatched
in waves that keep 64 in flight, instead of sequentially — same calls, same
cost, real concurrency. Provider rate limits permitting; where a tier caps
below 64, the achieved in-flight count is recorded in the report.

That evidence, though, comes from an **embeddings** API — a different service
class with different batching economics — so 64 was **provisional** until
measured where it would be used. The dry run (§13) ran the sweep
({16, 64, 128}) on the rehosted row (Llama-3.3-70B via Together, k=12 per
arm, achieved in-flight peak equal to the requested burst in every arm): no
arm separates from the row's own `same` under the CI-overlap rule
(`exact_generation` 0.378 / 0.448 / 0.399 against `same` 0.452
[0.382, 0.523]). Burst size does not change what the condition detects on
this service class, so **the panel-wide burst is frozen at 64** — continuous
with the representation-layer evidence base — and rows measured at 64 are
comparable on `conc`. The frozen number is also the input to the run-cadence
and cost decision.

`time` is measured against the previous scheduled run: pairwise agreement
between this run's `same` batch and the last run's, on identical prompts and
protocol. On a row's first run the condition reads `pending` rather than
inventing a same-day proxy — and the row publishes as a draft without an
ARI-D value until it resolves (§7's headline rule). Every `time` comparison records both runs' version
metadata (`system_fingerprint` and its equivalents), so a reader can tell drift
under a disclosed backend change from drift under a silent one — which is half
the point of the condition.

Headline core is `{proc, conc, time}`; `same` is the control every condition is
read against, using the CI-overlap classification the Matrix already uses — a
condition above its own `same` with non-overlapping intervals is a noise-floor
artifact, never an improvement. For `vendor` rows with no determinism controls,
a low `same` is itself the finding (in-session nondeterminism at temperature 0),
and the row's conditions are read relative to it rather than to a perfection the
provider never promised. The representation panel already found, for one API,
that the drift is per-call rather than connection-level — `proc` statistically
indistinguishable from its own `same`. This bench expects to reproduce that
reading for several rows, and reports it honestly as ≈ `same` rather than
treating a null as a failed condition.

## 7. Detectors

Reported per condition under the multi-detector schema (`detectors.<name>`),
each with `criterion_type` and `bytes_of_reference_state`:

| detector | type | what it reads | needs |
| --- | --- | --- | --- |
| `exact_generation` | exact | share of repeat pairs whose completions are byte-identical | any API |
| `first_divergence` | interval | mean normalized position of the first differing byte (1.0 = never diverged) | any API |
| `topk_logprob_overlap` | interval | mean Jaccard of top-k logprob sets per step, until divergence | logprobs |
| `reference_match` | exact | share of completions byte-identical to the fp32 reference | `rehosted` rows |
| `reference_first_divergence` | interval | mean normalized position of the first byte differing from the fp32 reference | `rehosted` rows |

The graded detectors carry `criterion_type: interval` — bounded overlap
measures in `[0, 1]`, which is precisely what the schema's `interval` means.
`threshold` was the alternative and is rejected deliberately: a threshold
verdict needs a calibrated null floor before it can be read, and the
null-relative-effects analysis shows `same` does not reliably provide one. An
interval needs no calibration to be read off.

Divergence is measured in **bytes, not tokens**. An API returns text; every
provider tokenizes differently, and one major vendor's tokenizer is not even
public — a token-based index would either be incomparable across rows or would
borrow one vendor's tokenizer to measure the rest. The first differing byte,
normalized by completion length, is universal, deterministic, and assumes
nothing. Precisely: for a pair of completions, `d` is the index of the first
differing byte; when one completion is a prefix of the other, `d` is the
shorter completion's length; the pair's value is `d / max(len_a, len_b)`, and
byte-identical completions read `1.0`. The normalization has a known
confounder — longer completions mechanically leave more room to diverge late —
so the dry run (§13) checks the correlation between completion length and the
normalized value per prompt bucket; if length dominates the signal, the
detector is redesigned before any paid panel run.

**How the reference pair reads — the honest-serving ceiling.** Providers
serve open models in half precision (bf16 is the industry default; the
checkpoints ship in it), and the decoding-reproducibility measurement
(`experiments/decoding-reproducibility/RESULTS.md`) says what that does to an
exact match against an fp32 reference: under *faithful* bf16, ~0.65–0.93% of
steps flip a token, and autoregressive compounding turns that into 20–35% of
whole generations differing at measured lengths (75% of completions survived
48 tokens; fewer survive 128). So **a faithful provider is expected to sit
well below `1.000` on `reference_match`, and that reading is not evidence of
infidelity.** The quantization signature is different in kind, not in degree:
under int8 the same measurement kept **zero** completions byte-identical and
first divergence collapsed from token ~38 to token ~3.5. That is why the
graded companion exists — the same research lesson the headline follows
(exact answers the binary question, graded separates regimes) applied to the
reference axis: `reference_first_divergence` separates faithful half-precision
(late, high) from quantized (immediate, near zero) robustly to completion
length, where the exact share alone would make honest rows look guilty.

Each `rehosted` row also records **`declared_precision`**: what the provider
publicly states it serves (some do — several "turbo"-class endpoints are
documented fp8; custom-silicon providers describe their own numerics). The
reading is two-axis: a low reference pair **with** a matching declaration is
consistent with what the provider advertises and reads as such; a low pair
with **no declaration** is the case the notification protocol (§11) exists
for. The bench never claims to identify *what* is being served — only whether
the output is consistent with faithful serving of the published weights under
mainstream arithmetic, read against what the provider says.

`reference_match` is a **diagnostic**, not folded into the headline: `vendor`
rows cannot have it, and folding it in would break cross-class comparability —
the same reasoning that keeps `prec` out of the ARI-R headline. It is also the
single most newsworthy number this bench produces, which is exactly why it only
runs where it cannot lie: **only against raw completion endpoints**, with the
model's official prompt template applied by us. A chat endpoint wraps the
prompt in the provider's own template, and at temperature 0 a one-byte template
difference produces a different completion from identical full-precision
weights — measured through a chat endpoint, this detector would read template
hygiene and report it as infidelity. Where only chat endpoints exist, the cell
reads `untested`; replicating each provider's template in the reference
generation is the v0.2 path to wider coverage.

**Headline: `ARI-D = mean(exact_generation over the three core conditions)`.**
Same shape as ARI-R's headline, same scorer invariant, same `1.000` reading. It
is never computed over a subset: while any core condition is `pending` (a row's
first run has no `time`), the row publishes in the Matrix as a **draft** —
conditions visible, the pending one marked — with no ARI-D value and no rank.
It enters the ranked column when all three conditions exist; an average over
two conditions and an average over three do not belong in the same sorted
column. The graded detectors are reported beside the headline and never
averaged into it.

## 8. Statistics, before any money is spent

The harness-power analysis showed our earlier instinct (few cases, few repeats)
would have missed a 20-point effect nine times out of ten, and that repeats
sharpen the control more than cases do. The same discipline applies here: before
the first paid panel run, a simulation over (n_prompts × k) grids must show the
design detects the effect sizes we claim to care about (a `time` drop of 0.10 in
`exact_generation`, say) at acceptable false-positive rates. Intervals are
bootstrap over prompts; the Δ-vs-`same` classification is the conservative
CI-overlap test already documented for the Matrix.

**Run (`experiments/arid-power-sim/RESULTS.md`), and the design holds where
this section makes promises**: Δ=0.20 detected always, Δ=0.10 at 0.89–0.96
power under bimodal deviations at noisy floors, false positives ≤1% across
all 252 cells, and k=4 would not have sufficed — the frozen k=8/12 stands
unadjusted. Two caveats the reader gets in writing: a `≈ same` reading on
`exact_generation` does **not** exclude a ~0.10 drop concentrated on few
prompts under compounding deviations (power ~0.53 there; more repeats do not
move it, only more prompts do — 100 is a working minimum, not a comfortable
ceiling), and five-point drops are not a claim this bench makes (power ≤0.33).
The graded diagnostics beside the headline are the detectors with a shot at
the concentrated regime; whether v0.2 adds a paired per-prompt test for it is
an open methodology question, flagged with the evidence.

## 9. Schema, scorer, verification

- Reports carry `layer: "decoding"` and `input_set: "ARI-D-Bench-v0.1"`; the set
  joins `INPUT_SETS` in the scorer. The detector block needs no schema change —
  the multi-detector shape was built for exactly this.
- Scorer invariants adapt by layer: the ARI-equals-mean check reads
  `exact_generation` instead of the semq HER; pin, attestation and signer-
  registry checks are identical.
- Reports are signed with the project key (`alias/semq-ari-attestation`),
  manifests binding the prompt set **and the response transcripts**, and verify
  with the existing standalone verifier. Rows classify self-reported/attested
  exactly as Representation rows do.

## 10. Panel and cost, v0.1

Starter panel: OpenAI, Anthropic, Gemini, Mistral (vendor class), plus two
open-weight models on three rehosting providers each (rehosted class) — 10
rows. Rehosted models are chosen by rule, not by name, because provider
catalogs rotate under any name this document pins: **an open-weights model
currently served by at least three independent rehosters, its exact Hugging
Face revision pinned at prompt-set freeze time.** Current candidates fitting
the rule: Llama-3.3-70B and Qwen2.5-72B. (The rule already earned its keep:
this document's first draft named Llama-3.1-70B, which at least one panel
provider no longer serves.)

Provider capability, verified against public API documentation at proposal
time and re-verified in the dry run:

| provider | raw completions | logprobs | consequence |
| --- | --- | --- | --- |
| Together | yes (`/completions`) | top-5 + echo (top-20 rejected live, dry run) | full detector coverage; `topk` at k=5 |
| Fireworks | yes (`/v1/completions`) | yes + echo | full detector coverage |
| Groq | chat only | not documented | `reference_match` reads `untested` |
| DeepInfra | probable (OpenAI-compatible) | unconfirmed | confirm in dry run |
| OpenAI | n/a (vendor) | top-20 | `topk_logprob_overlap` runs |
| Anthropic | n/a (vendor) | none | sequence detectors only |

Per row per full run: 100 prompts × 36 completions × ~128 output tokens ≈ 460k
output tokens — roughly $2–8 per row at current list prices, $50–80 per full
panel. Weekly cadence lands around $250–350/month. The fp32 references for the
rehosted class are one GPU session per model release (the same g6e box the ARI-D
experiments use), reused until the weights change.

## 11. Provider notification and response

The bench publishes rows that name companies, so it owes them the process it
would want in their place — and its own receipts make that process cheap to
offer.

- Before a provider's row first publishes, and whenever a row's reading
  worsens materially between runs, the provider is notified with the complete
  receipt: the report, its signed manifest, and the one-line command that
  re-verifies both without installing our stack.
- The provider has a fixed response window (seven days). A response is
  published verbatim beside the row. Silence is noted as "no response", not
  interpreted.
- There is no pre-publication veto. The row publishes on schedule with or
  without a response; the response can contest, contextualize, or point to a
  disclosed change — the recorded `system_fingerprint` history makes "we
  shipped an announced upgrade" a checkable claim rather than a shrug.
- A demonstrated measurement error is corrected and the correction is
  published with the same prominence as the original reading. The retraction
  discipline that built this project's credibility applies to its bench.

## 12. What v0.1 does not claim, and open questions

- Nothing above temperature 0. Whether the protocol separates infrastructure
  drift from sampling noise at temperature > 0 is the harder claim, still open.
- A drift that never changes any output is invisible here by construction —
  and one is documented. Under TF32, three production models emitted
  byte-identical completions (100% of tokens, 48 of 48 generations) while the
  research instrument read a changed logit code at essentially every step.
  **This bench would score that serving change as perfectly stable.** Seeing
  below the argmax requires logits, which is the white-box instrument's
  territory; an audit claim about the computation, as opposed to the output,
  needs the instrument, not this bench.
- Beyond the TF32 class, no coverage claim. Whether real serving drift is ever
  confined where sequence-level detectors are blind remains the open question
  it was for the research instrument. Answering it means running the
  full-logit-vector probe beside these detectors on self-hosted models —
  research-program work, in `experiments/`, not a bench detector. If
  tail-confined drift is ever shown in the wild, that finding earns a
  detector; until then the bench does not carry one on faith.
- A `vendor` row with a low score is a reproducibility reading, not a quality
  judgment. The bench measures whether you get the same thing twice, and says
  nothing about whether it is good.
- The prompt set is public and frozen, so a provider could in principle serve
  cached completions of exactly these prompts and read perfectly stable. v0.1
  accepts that risk and watches for its signature: byte-identical outputs
  across a change of `system_fingerprint` is the red flag — a backend that
  changed but answers identically to the old bytes is caching, not stability.
  If gaming ever moves from possibility to suspicion, the designed escalation
  is commit-reveal: publish the next set's content hash before measuring and
  the prompts themselves after, so nothing can be pre-cached while the pin
  stays verifiable.

## 13. Rollout

1. Power simulation over the (n_prompts, k) grid; adjust §6 repeats if needed.
2. Freeze `data/arid-bench-v0.1.jsonl`, publish its content hash.
3. Dry run against one vendor and one rehosted provider; fix the harness.
   The dry run also settles three §4/§6/§7 calibrations: the `conc`
   burst-size sweep ({16, 64, 128}) on the open-model rehosted row that
   freezes the panel-wide burst, the `first_divergence` length-confounder
   check (correlation between completion length and the normalized value,
   per prompt bucket), and the **set-discrimination check** — the `same`
   floor per bucket, plus the top-2 margin distribution where logprobs
   exist. If `same` saturates at 1.000 across all buckets on both subjects,
   the set cannot separate drift from stability and gets enriched toward
   decision-dense prompts *before* the panel runs, as a v0.1 amendment
   (re-freeze, new hash) rather than waiting for v0.2.
4. fp32 references for the two open models (one GPU session). The same
   session also generates **exploratory** bf16 and fp16 reference passes
   (the model is already loaded; marginal cost is near zero). These are
   dry-run data, not published detectors: they exist to *test* the
   hypothesis that a faithful half-precision serving matches a
   same-precision reference more closely across stacks — a claim no
   experiment currently supports, and cross-stack same-precision agreement
   is exactly the kind of claim this program has had to retract before.
   If the data confirms it, a same-precision reference detector enters
   v0.2 with evidence; nothing about it is assumed in v0.1.
5. Full panel, signed; rows land in the Decoding tab through the standard
   scorer gate. The tab stops being empty the same way Representation started:
   measured, pinned, attested.

## 14. Provenance of this design

This document assembles prior work rather than inventing. The layer definition
and its prerequisites come from the ARI decomposition proposal
(`docs/proposals/ari-decomposition.md`). The metric lessons — exact and graded
answer different questions, ship the margin, power before spend — come from the
decoding experiments, their baselines, and the harness power analysis, as
recorded in `docs/findings.md`. The multi-detector report shape, the
Δ-vs-`same` classification and the `untested` cell discipline are the
leaderboard's existing machinery. The pin, the signing key and the independent
verifier are the attestation stack already live on the board. What is new here
is the hosted-API protocol: the two subject classes, the sequence-level
detectors, and the frozen prompt set.
