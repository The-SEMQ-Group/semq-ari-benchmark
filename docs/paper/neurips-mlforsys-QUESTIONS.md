# Open questions on the ML for Systems draft

Companion to [`neurips-mlforsys-ari.md`](neurips-mlforsys-ari.md). Everything here is a
judgment call I made while drafting, flagged so you can overrule it during review.
Inline `<!-- TODO -->` comments in the draft mark the spots where a question is local
to one sentence.

## Venue and format — resolved against the CFP (mlforsystems.org)

1. **Workshop fit: confirmed, with a framing lever.** The 2026 call explicitly solicits
   "best practices, methodologies, benchmarks, datasets, simulators, and evaluation
   frameworks that improve rigor, reproducibility, reliability, and trustworthiness",
   and lists "Reliability, robustness, safety, and trustworthiness of ML-driven
   systems", "Benchmarking and evaluation methodologies", and "Systems for LLMs,
   multimodal models, and agentic applications" as areas of interest. One nuance: the
   reproducibility bullet is written about rigor *in ML-for-Systems research*, while
   this paper measures reproducibility *of deployed ML serving infrastructure*. The
   safest read of the call for us is the AI-infrastructure and agent-reliability
   framing, which the rewritten intro now leads with.
2. **Page limit: 4 pages, STRICT, excluding references and appendices.** NeurIPS 2026
   format, PDF, via OpenReview. **The draft is over.** At 3,065 body words plus four
   tables it lands near 5.5 pages in the NeurIPS template; about a third has to come
   out of the main body. An optional appendix is allowed, and reviewers are not
   required to read it, so material moved there should be support, never a load-bearing
   claim. See "Cutting to four pages" below.
3. **Double-blind: no.** "Submissions do not have to be anonymized." The SEMQ name, the
   repo links, and the KMS/signer details all stay, and §2 "Verifiability" needs no
   anonymized-artifact rewrite. This is a net gain: the verifiability argument depends
   on naming a public signer registry a reviewer can actually check.
4. **Deadline: August 29, 2026 AoE, extended to Monday August 31.** The CFP's important
   dates say August 29; the submission instructions carry an extension to August 31
   midnight AoE (= 12:00 UTC on September 1). Today is August 29. Notifications
   September 29; workshop December 11 or 12. No formal proceedings, so submitting here
   does not block publishing the work elsewhere later.
5. **Still to do for format:** the draft is markdown and the submission must be a PDF in
   the NeurIPS 2026 LaTeX template. That conversion has not started.

## Cutting to four pages — done

The draft was 3,065 body words and measured near 5.5 pages. It is now cut and
built: **the body ends on page 4**, with references and appendices following.
Verify with `grep -o "newlabel{bodyend}.*" latex/ari.aux` after two pdflatex runs.

What came out of the main body, in order of size:

- §4 "What our own testing took away" → Appendix A. It is a retrospective on our
  own corrections, valuable but not load-bearing for any claim in the body.
- Most of the old §5 Limitations → Appendix B. The three limits that actually
  bound the claims (greedy-only decoding, single-machine serving conditions,
  operator self-attestation) stay in the body as a `\paragraph`, not a section.
- The NVFP4 influence-spread aside in §2 → Appendix D.
- The chunked-vocabulary control in §3.3 → Appendix C.
- Prose tightening throughout: abstract, §1, §2 metrics and verifiability, the
  §3.1 checks, the §3.2 mechanism and scope note, §3.3, §3.4, conclusion.

Nothing was cut from §3.1-3.4. All four results subsections, all four tables and
every number survive in the body.

## Build status

- `latex/ari.tex` is the source of truth. The markdown mirror is STALE as of the
  abstract rewrite.
- Builds clean with a plain `pdflatex ari.tex` (twice). The missing packages
  (`environ`, `trimspaces`, `lineno`) were fetched from CTAN and installed into
  `~/Library/TinyTeX/texmf-local/`; nothing is stubbed.
- `latex/neurips_2026.sty` is the official 2025 file with the package name,
  ordinal (40th) and year (2026) changed locally, because NeurIPS has still not
  published a 2026 Styles.zip. Swap in the official file when it appears; the
  geometry is unchanged, so the page count holds.
- Current measurement: 5 pages total, **body ends on page 4**, within the limit.

## Peer review response (implemented 2026-08-31)

Sixteen-point review from a colleague. Implemented in `latex/ari.tex`; body still ends
on page 4.

**Done, with new analysis:**

1. **HER cross-model comparability.** The objection was real: HER is
   $P(\text{code identical})\approx(1-p)^{2d}$, so it falls with dimension at fixed
   stability, and $\kappa$ varies over $2\times$. Rather than hedge, we tested it.
   Using the published per-model $(d,b,\kappa)$ we recomputed the panel as the
   noise-equivalent $\hat\sigma=(\bar H/\kappa\sqrt{2d/\pi})^{1/b}$, which divides both
   confounds out. **The ordering is unchanged.** Under raw $\bar H/d$ only the bottom
   two swap, and they are within 20% either way. This is a new result computed from
   the existing `Hbar` fields in the submissions plus `spec/fingerprints-v0.1.csv`;
   it is now Appendix G, cited from §2 and §3.1.
2. **Causal overreach.** Condition table header is now "varies / variation exposed",
   not "perturbs / isolates". `conc` exposes batching, composition, routing and
   queueing; `time` exposes time-associated drift with cause unobservable; `proc`
   is split by system class. §2 states that on a hosted API drift is attributed to
   the treatment, never to a backend mechanism.
3. **Pseudoreplication.** Confirmed from `ari/metrics.py`: the bootstrap resamples
   inputs only. New **Replication** paragraph in §2 main text says each condition is
   one execution pair, that intervals are input-set uncertainty conditional on that
   pair, and that one 76-hour comparison detects time-associated drift without
   characterizing it.
6. **Quantizer.** The reviewer was right that "2 bins" was wrong. `ari/probe.py` is
   QBIN `n=2` with symbols from edges $\{-s,0,+s\}$ — four symbols, about 2 bits per
   dimension — so the p99 scale $s$ does matter. Fixed. The law is
   $\mathbb{E}[\bar H]=\kappa\sqrt{2d/\pi}\,\sigma^{b}$: $b$ is the fitted **exponent**,
   not a slope absent from the equation, and $s$ is now defined.
5. **ARI-E estimator.** Verified against `experiments/harness-effect/RESULTS.md`: the
   estimator is an agreement gap $\Delta=D_{\text{cross}}-D_{\text{self}}$, and all four
   contrasts reproduce exactly. Importantly the old text was misleading: the "21.5-point
   outcome difference" is a **cross-condition disagreement rate**, not a success-rate
   difference, so nothing is being subtracted from a success delta. Now stated with the
   equation, plus Appendix E on assumptions. Causal language removed.
4. **Compounding experiment** relabelled a controlled stress test, with the Gaussian
   named as synthetic and the covariance/anisotropy/temporal-correlation gap stated.
7. Top-$K$ Gaussian formula deleted; §3.2 measures the same thing empirically.
8. "internal state" replaced throughout with output/downstream representation.
9. "third-party-verifiable" split into tamper-evident report integrity vs independently
   reproducible measurement.
10. EU AI Act / NIST sentence rewritten: frameworks "call for consistent, monitorable
    behaviour and some name reproducibility as a concern", none prescribes a metric.
11. "a different replica's vector" and "self-host and pin the precision" both struck.
12. §3.2 retitled "Recall@10 can remain unchanged under large representation drift";
    "6% mean cosine" is now "mean cosine similarity 0.938" (from the results table).
13. HER saturation acknowledged in §3.3 as the same fact reversed.
14. Appendix A reframed as a design tradeoff: ARI-D is a high-sensitivity diagnostic,
    top-rank margin is the cheaper monitor.
15. References [2], [3], [5], [6], [7] completed; [7] correctly attributed to Thinking
    Machines Lab with its batch-invariance argument stated.

**Not done — needs experiments we cannot run before the deadline:**

- **Replaying empirical API residuals through the multi-hop retrieval experiment.**
  This is the reviewer's "experiment I badly want" and they are right. Handled by
  labelling the result synthetic and naming the replay as the next experiment.
- **Multiple independent runs per condition**, especially repeated `time` gaps.
  Handled by disclosing the replication unit in main text.

**Left for you to decide:**

- **The name.** The reviewer proposes *Auditable* Reproducibility Index over *Agent*,
  same acronym, and the argument is good: the headline experiment is embedding APIs.
  Not changed, because "Agent Reproducibility Index" appears 17 times across the repo,
  the leaderboard, the spec and `CITATION.cff`, and renaming the paper alone would
  contradict the artifact it links to.
- **More related work.** The reviewer wants citations for floating-point
  non-associativity, deterministic numerical computation, GPU/TF32 numerics and
  benchmark variance. Not added: inventing plausible-looking citations is worse than
  leaving the gap visible.

## Section 2 review response (2026-08-31, second review)

Sixteen-point review of the protocol section. Implemented in `latex/ari.tex` using the
reviewer's inline-edit text where supplied. Every fact below was verified against the
`semq`, `semq-research`, or `semq-ari-benchmark` source rather than assumed.

**A retraction first.** The earlier Appendix G claim that the sigma-hat ordering
"is unchanged" was wrong — my inversion fed bits into a law fitted on the fraction of
changed code positions, omitting the per-model normalization. Recomputed correctly,
**voyage and cohere swap**: cohere's noise-equivalent drift (3.7e-4) is slightly below
voyage's (5.0e-4), so HER's three-fold gap between them is mostly the exact-match
penalty on cohere's 1536-dim code. Section 3.1, Appendix G, and the conclusion now say
this. It makes the reviewer's comparability objection (point 1) load-bearing, not
hypothetical.

Verified facts now stated in the paper:

- **Quantizer** (`semq/src/core/semq_quant_scalar.c`): four-level symmetric scalar
  quantizer; per coordinate, sign bit + magnitude bit with thresholds {−s/2, 0, +s/2},
  s = p99(|x|) over the reference batch. The old "edges {−s, 0, +s}" wording came from
  the CI **mock** in `ari/probe.py`, not the canonical operator — a real error this
  review caught. The ±s saturation rule is removed as redundant: the indicator form
  already implies it.
- **Hamming semantics**: the two bits per dimension are independent semantic
  indicators (sign; magnitude bin), so bit-level Hamming = sign changes +
  magnitude-bin changes — not an arbitrary label assignment. (The SDK's own compare is
  symbol-level; ours is the indicator sum.)
- **Calibration model** (`semq-research/L1_10`): eps ~ N(0, (sigma * mean-norm)^2 I),
  added to raw embeddings, explicitly NOT re-normalized; sigma is the per-coordinate
  noise SD as a fraction of the mean representation norm. The response law is fitted on
  **rho = fraction of changed code positions under the packed comparison** (four
  coordinates per byte), approx 4*Hbar/d at low drift — NOT Hbar/(2d) as the reviewer's
  inline edit guessed; adopting that would have made the published kappa wrong by ~8x.
- **b** is model-specific and used per-model in the inversion; 0.979 ± 0.028 is
  mean ± SD over the 13 per-model fits (recomputed from `spec/fingerprints-v0.1.csv`).
- **Bootstrap** (`ari/metrics.py`): plain nonparametric bootstrap over inputs,
  resampling whole input rows. "Block-bootstrap" language removed.
- **ARI-D** (`experiments/decoding-reproducibility/run_matrix.py`): teacher-forced —
  every condition is fed the reference token sequence, so per-step logits respond to
  identical prefixes; free-running generations are the separate outcome-divergence arm.
- **Verifier** (`ari/verify_report.py`): stdlib-only imports, so "no third-party
  dependencies and does not import the ARI implementation" is literally true.

Structural decisions:

- **The headline (reviewer's point 1)**: resolved as the reviewer's option 3 in
  substance. ARI_HER is defined as strict exact-repeatability over the externally
  inducible core; the paper states raw HER is not by itself a normalized cross-model
  measure and pairs it with sigma-hat. The "comparable core" phrase and the "lower
  bound on true drift" sentence are gone; conditions table is now
  "varied condition / variation potentially exposed" with mach and lib restored.
- ARI-E's Delta estimator is defined with its equation; exchangeability and scope
  assumptions live in Appendix E; "attributable/causal" language removed.
- The 4-page fit was recovered after the fuller Section 2 by: folding §3.3's intro
  into the stress-test paragraph, merging the three bf16 decoding rows into one range
  row, prose trims across §2–§5, small heading-skip reductions (−7/−7/−9pt) at the
  page-4 boundary, and tighter display spacing. Body ends on page 4 (verified via
  the bodyend label); no overfull boxes.

Not done (needs experiments): replaying empirical API residuals through the multi-hop
walk; multiple independent runs per condition, especially repeated time gaps. Both are
disclosed in the text as limitations/next experiments.

## The rho unit question (2026-08-31, third review)

The reviewer challenged the definition of rho and instructed: check what quantity b and
kappa were actually fitted against before repairing prose. Done, conclusively:

- `semq` binding `batch_encode` returns packed bytes, `(n, d/4)` at n_bins=2.
- The sweep's `_per_row_hamming` compares those arrays elementwise, so the fit target
  is the **fraction of packed code bytes that differ**.
- Numeric confirmation from `experiments/drift-sensitivity/results/sensitivity_curves.csv`:
  bge-large at sigma=1e-6 reads hamming_mean=8.0e-5; byte-level theory predicts ~4e-5
  (ratio ~2, matching the recorded "1.9x consistent overestimate"), while bit-level
  (~5e-6) and coordinate-level (~1e-5) are off by 16x and 8x. Only the byte-fraction
  reading is consistent with kappa in [1.48, 3.06].
- **Repo bug found and fixed**: `experiments/drift-sensitivity/results/README.md`
  labeled `hamming_mean` as "fraction of SEMQ code bits flipped". It is the packed-byte
  fraction. Corrected.

Resolution in the paper — no refit needed, because sigma-hat is exactly invariant to
the unit change: byte-fraction ~= 4x coordinate-fraction at low drift, so restating the
law at coordinate level with kappa_c = kappa/4 leaves (rho/kappa*sqrt(2d/pi))^(1/b)
unchanged (b is unit-invariant in a power law; the 4s cancel). Section 2 now defines
rho as the **coordinate change fraction** — the unit the reviewer asked for — with the
principled low-drift conversion rho ~= Hbar/d (a changed coordinate typically flips one
of its two indicator bits: sign near zero, magnitude near +-s/2; exact bounds
Hbar/2d <= rho <= Hbar/d). kappa is now quoted coordinate-level: [0.37, 0.76] across 13
architectures, [0.37, 0.66] for the five APIs. Appendix G notes the registry stores 4*kappa
(packed convention) and that sigma-hat is convention-invariant; its sigma-hat values are
unchanged.

Follow-up owed to the repo, not the paper: `spec/fingerprints-v0.1.csv` stores kappa in
the packed convention without saying so; a v0.2 should either store coordinate-level
kappa or name the convention in the header.

## Restructure and recovery (2026-08-31, late)

Everything after `\appendix` was accidentally deleted from `latex/ari.tex`. Recovered:
Appendices A-D and the reproducibility statement came back verbatim from the
pre-review backup in the session scratchpad (`ari-pre*.tex`); E, F and G were written
this session and were reconstructed from their source edits. Nothing was taken from
`ari-old.pdf` — no PDF text extractor is installed on this machine — but that file is
still on disk at 17:01 if you want to diff by eye.

Appendices are now `\label`/`\ref`-driven rather than hardcoded letters, so reordering
can never silently break a cross-reference again. Current order:

| | appendix | contains |
| --- | --- | --- |
| A | Probe and calibration specifics | perturbation model, bits-to-coordinates conversion, registry convention, sensitivity spread |
| B | Verifiability details | manifest contents, what the signature covers, verifier independence |
| C | Outcome-layer reproducibility (ARI-E) | the moved §3.4, four-contrast table, estimator assumptions |
| D | Normalized panel ordering | sigma-hat recomputation |
| E | Float-level residuals | |
| F | Decoding controls | |
| G | What our own testing took away | |
| H | Remaining limitations | |

Body changes: §3.4 moved out wholesale, leaving a one-sentence pointer so the ARI-E
layer is not orphaned after §2 introduces it; Verifiability cut to two sentences;
calibration specifics moved to A; §3.1 normalization and §3.2 prose compressed against
their adjacent tables. Body is now 1,892 prose words and ends on page 4; appendices
run to page 7.

**Scope note for review:** the body no longer contains the agent-reliability result,
which is the part of the paper matching the CFP's agentic bullet most directly.
Reviewers are not required to read appendices. If the paper reads thin against the
call, Appendix C is the first thing to promote back.

## sigma-hat: intervals now, direct rho in v0.2 (2026-08-31)

A reviewer objected that ARI-sigma-hat exists to fix HER's comparability problem yet
itself rested on the approximation rho ~= Hbar/d, and that Appendix D called the
bottom-two gap "comparable to the method's own approximation error" while also saying
"their order reverses." Those two statements contradicted each other. The reviewer was
right; the second was an overreach I introduced.

Fixed without a refit, by propagating the exact bound Hbar/2d <= rho <= Hbar/d instead
of assuming the low-drift limit. Every reported sigma-hat is now an interval:

| model | sigma-hat interval |
| --- | --- |
| gemini-embedding-001 | 0 |
| text-embedding-3-large | [0.53, 1.12] x 10^-5 |
| mistral-embed | [2.84, 6.07] x 10^-5 |
| voyage-4-large | [2.46, 4.99] x 10^-4 |
| embed-v4.0 | [1.84, 3.68] x 10^-4 |

Every adjacent pair among the top three is disjoint, and mistral-embed is disjoint from
both remaining systems, so those separations survive the conversion outright. The bottom
two overlap, so the paper now states that **sigma-hat does not resolve an ordering
between voyage-4-large and embed-v4.0**, and that the point-estimate reversal is not
evidence. Section 3.1 was corrected to match.

This is strictly stronger than the point estimate it replaces: the bound is exact, not
approximate, so no reported comparison depends on the low-drift assumption any more.

**Still owed to the registry (v0.2), unchanged in priority:** serialize
rho = (1/d) * sum_j 1[Q(x_j) != Q(x_j')] directly from the coordinate codes and refit
(b, kappa) against it. That collapses every interval above to a point and removes the
conversion from the method entirely. It needs the full 13-encoder sweep (SDK, embedding
caches, five paid APIs), so it is not a deadline-night change. Note the raw code
matrices are not in the repo -- submissions carry audit hashes, not codes -- so the refit
means re-running, not re-analyzing.

## Reviewer gaps answered from existing data (2026-08-31)

A reviewer named five weaknesses. Two already had evidence in the repo that the paper was
not using; three do not and are genuine gaps.

**Answered — condition replication.**
`experiments/deployed-agent-panel/artifacts/openai_proc_pilot_n1000.json` (from
`ari/tools/proc_pilot.py`) ran text-embedding-3-large over the same 1,000 inputs in three
extra fresh-context realizations. With the panel run that is four realizations:

| condition | HER per realization | sd | within-run CI half-width |
| --- | --- | --- | --- |
| proc | 0.851, 0.856, 0.846, 0.862 | 0.007 | 0.022 |
| same | 0.849, 0.837, 0.870, 0.859 | 0.014 | 0.022 |

Realization variance for `proc` is about a third of the input-set interval, so the
single-realization estimate is stable on that provider. Now in Section 2 and Appendix E.
It does **not** generalize: `conc` was never repeated, so voyage's 0.209 burst -- the
reviewer's own example -- is still one load event; `time` was repeated only at a different
gap (mistral 0.753 at 19h vs 0.554 at 76h).

**Answered -- baselines against simpler statistics.**
`experiments/decoding-reproducibility/BASELINES.md` is a three-axis comparison against
top-2 margin, max|dlogit|, L2, top-20 overlap, KL, JS and token flip. The paper previously
cited only the 86%/8-byte concession. Now also in Appendix I: KL and JS are second order
so they are not instruments at small sigma (KL reads -5.1e-10 at sigma=1e-4, wrong sign,
where the probe reads 1.33e-4); top-20 change is sub-linear; token flip is a threshold
detector; and only the probe and the top-2 margin are bit-exact under regrouping, while KL
moves 4.6% and JS 363%. That is the crisp "why quantize" answer, and it honestly does not
separate the probe from the top-2 margin.

**Not answered, no data exists:** replaying measured API residuals through the trajectory
experiment; repeated `conc` and same-gap `time` realizations; a retrieval-layer baseline
sweep as systematic as the decoding one. All three need runs, not analysis.

## Content decisions I made

5. **Included ARI-E (§3.4), which the earlier short draft omitted.** Reasoning: the
   scaffold-vs-model result with the self-consistency control is the most
   systems-community-shaped finding we have, and it broadens "reproducibility" beyond
   embeddings. Cost: about a third of a page. Say the word and it comes out.
6. **Attestation status.** Main now registers the KMS Ed25519 key in
   `spec/signers.json` and PR #15 attested all 13 submissions, so I dropped the old
   draft's "no durable signing key is registered, so none is attested" limitation and
   replaced it with the narrower "operator attests its own reports" caveat. But
   `leaderboard/README.md` still says every row reads `self-reported`. Which is
   current? The paper should match the repo state at submission time.
7. **TF32 mantissa.** The earlier short draft says TF32 has a "~19-bit mantissa". TF32
   is 19 bits *total* (1 sign, 8 exponent, 10 mantissa). I wrote "~10-bit mantissa" and
   left a TODO at the spot. Please confirm the intended phrasing.
8. **Dropped the Voyage/Anthropic-docs remark.** The panel RESULTS note that
   voyage-4-large is the embedder Anthropic's docs recommend. Pointed, but it reads as
   a jab in a peer-reviewed submission and adds nothing to the measurement. Restore it
   if you want it.
9. **The 11.5% trajectory result (L3_09) lives in semq-research**, which I believe is
   private. The reproducibility statement says everything is in "the accompanying
   repository". Either migrate L3_09 into semq-ari-benchmark before submission or
   scope the statement. Same question, milder, for the 13-architecture power-law fit
   (b = 0.979 ± 0.028), which comes from the whitepaper's cross-repo aggregation.
10. **Open-SWE-Traces credit.** ARI-E analyzes NVIDIA's public dataset; the rollouts
    are theirs, the analysis is ours. §3.4 names the dataset, but check the dataset
    license permits this use and decide whether it needs more than a citation.
11. **Regime-discrimination table freshness.** The GPU rows (TF32 on/off) in §3.2 come
    from the updated run recorded in `experiments/README.md` and `docs/findings.md`;
    `experiments/regime-discrimination/RESULTS.md` itself still says GPU conditions are
    deferred. The numbers I used (0.5267, 0.9992, 47%) are consistent across
    findings.md, the whitepaper, and the experiments index, but that RESULTS.md should
    be brought up to date so a reviewer following the repro trail doesn't hit a
    contradiction.
12. **"Batched" decoding condition omitted.** ARI-D also measured batch-composition
    drift (19-67% of steps across models). I left it out for space and because the
    per-model spread complicates the story. It could replace the TF32-on rows for
    Qwen/Mistral if you'd rather show breadth.

## Numbers: every figure in the draft traces to a repo artifact

| claim in draft | source |
| --- | --- |
| API panel 0.169-1.000, per-condition table | `experiments/deployed-agent-panel/RESULTS.md` |
| cache probe (150 nonce inputs, 0.807 control), 3/30 float-level, ~2,300/3,072 dims | `docs/methodology.md` |
| self-hosted 1.000 / bf16 cliff ≤ 0.094 | `experiments/deployed-agent-panel/RESULTS.md` |
| SciFact table, TF32 47%, 0.9992, bf16/int8 rows | `docs/findings.md` §2, `experiments/README.md`, `experiments/regime-discrimination/` |
| isolation 2x2, 0.749 | `experiments/gpu-determinism/RESULTS.md` |
| walk: 11.5%, 8.6x, σ grid | semq-research `docs/findings.md` (L3_09) |
| decoding table, 26-45x amplification | `experiments/decoding-reproducibility/RESULTS-production-models.md` |
| ARI-E contrasts, 21.5→8.9, 59-74% | `experiments/harness-effect/RESULTS.md` |
| top-2 margin 86% / 8 bytes | `experiments/decoding-reproducibility/BASELINES.md` |
| rank-profile 28.7x enrichment, 100.00% token agreement | `experiments/drift-rank-profile/RESULTS.md` |
| probe-verifiability sweep, 91% near-tie | `experiments/probe-verifiability/RESULTS.md` |
| b = 0.979 ± 0.028, κ ∈ [1.48, 3.06] | `docs/whitepaper.md` §(universal form), `experiments/drift-sensitivity/` |
