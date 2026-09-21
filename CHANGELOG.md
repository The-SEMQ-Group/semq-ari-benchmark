# Changelog

All notable changes to ARI (the spec, the benchmark, and the reference implementation).
This project follows [semantic-ish versioning](https://semver.org/) for the **spec** (v0.1,
v0.2, …); the leaderboard tracks a preview label until the spec is frozen.

## [Unreleased]

### The time condition carries its evidence
- `spec/report-schema.json` gains an optional `time_evidence` object on a
  condition result (`$defs/timeEvidence`): UTC start and end of the baseline
  and comparison encodes, `gap_hours` (greater than 24), the probe id and
  frozen scale, the input hash, model and tokenizer revisions, precision,
  hardware, SDK version, harness commit, and a `deviations` list. Optional, so
  reports signed before it existed still validate; `ari_version` stays `0.1`.
- `ari.verify_report.check_time_condition` rejects a `time` result without the
  block, with a gap of 24 hours or less, with a `gap_hours` that disagrees with
  its timestamps, with a moving ref (`main`, `unknown`) as a revision, or whose
  block differs from the report's own input hash, probe id or precision. Each
  violation names the field. `python -m ari.check_evidence <report>` runs schema
  validation and the same function; it is the conformance gate, so a `time`
  cell without the block fails unless `--no-require-time-evidence` is passed.
  The standalone verifier checks a block that is present and fails on a
  violation; a cell without the block prints a warning and passes, so reports
  signed before the block existed still verify without re-signing.
  `--require-time-evidence` turns that warning into a failure.
- `ari/tools/capture_time_baseline.py` records the block's baseline half in
  `meta.json`, refuses a comparison whose revision, precision or SDK version
  differs from the baseline, builds the `time` cell with `time_evidence`, and
  with `--report`/`--out` merges it into an existing report and recomputes ARI.
  `--agent st` captures a local sentence-transformers model on CPU with pinned
  threads. A baseline holding only the pre-existing `unix_ts` is accepted with
  a recorded deviation. Precision is read back and recorded, never changed.
- `ari.report.semq_condition_entry` and `ari.report.core_ari` are factored out
  of `build_report` so a tool can build one cell and recompute the aggregate.
- `SentenceTransformerAgent` accepts a `revision` and reports its `precision()`.
### SciFact codes committed beside the results
- `experiments/regime-discrimination/results/codes/` holds the packed QUANT
  codes for the reference and the six CPU conditions, with the calibration
  scale, bit layout and file hashes in `MANIFEST.json`. `export_codes.py`
  writes them (SDK required) and `--check` recomputes HER from them with
  `ari.code_metrics` alone, so the code-level rows can be verified without
  the SDK. Verified equal to the committed table on every condition.

### The mock probe is gone
- `ari/probe.py` binds to the SDK only. The development stand-in quantizer is
  removed: its magnitude edge and byte layout differed from the canonical
  probe, so its HER and byte rates were comparable with nothing, and a
  stand-in that matched the SDK would be the reimplementation CONTRIBUTING.md
  forbids. `load_probe(X)` takes no backend and raises a clear error without
  the SDK. The `--probe-backend` flag is gone from `ari.run`, `run_report`,
  `selfhosted_pilot` and `proc_pilot`, and `proc_pilot` no longer falls back
  to a stand-in silently. The mock *agent* stays; it fakes vectors, not codes.
  
### CI says when the probe was not tested
- A new `sdk-tests` job fails by name until `CODEARTIFACT_ROLE_ARN` lets CI
  install the SDK; with it set, the job installs `semq`, runs the suite and
  fails if any test still skips for want of it. The public `pytest` job
  prints how many SDK-gated modules skipped. A green check no longer looks
  the same whether or not the probe ran (SEM-52).
  
### The leaderboard refresh has no mock mode
- `refresh_leaderboard.py --mock` is removed, with its fake agent and its
  numpy stand-in quantizer. The stand-in rounded to 256 uniform levels and
  shared nothing with the canonical probe, so the rows it produced were
  comparable with nothing; a stand-in that matched the SDK would be the
  reimplementation CONTRIBUTING.md forbids. Every code the tool writes now
  comes from the SDK.

### The operator is called QUANT
- Prose, docstrings and comments say QUANT, the name the SDK has used since
  1.5. QBIN remains only in a historical note in `semq_compat.py`, which now
  requires `semq` 1.5.1, and in the record of the rename itself. The
  drift-sensitivity README's reference snippet now calls the harness probe
  instead of an SDK function that never existed under either name. Editorial
  only; no probe parameter, hash or scoring rule changes.

### The SDK is the authority on layout and calibration
- `run_pilot.py` took the packed layout, the symbol unpacking and the
  calibration percentile from the core instead of reimplementing them. The
  inline unpacker read each byte's symbols high-order first: aggregate rates
  came out right, because the permutation is the same in both operands, and
  every changed-coordinate index came out wrong.
- A probe wider than one context is chunked again, which the rewrite had
  replaced with a refusal. Calibration stays global — the core takes the
  percentile over the whole buffer, so calibrating on the array reshaped to
  the chunk width gives the same float32 scale a single wide context would,
  and that one scale is fixed across every chunk. Chunk comparisons are joined
  with `CodeComparison.concatenate`, so counts are summed rather than rates
  averaged and `changed_coordinates` stays usable. The reported score is
  bit-identical however the probe was split, which is tested at five chunk
  widths.
- `semq_compat` no longer claims `calibrate` and `numpy.percentile` give
  bit-identical codes. They differ when the percentile falls between two
  float32 values.
- PROTOCOL.md §0.5 records that the committed pilot cannot be checked against
  the current scoring: its input cache is not in this repository and the scale
  it recorded is not what either path produces on the cache that is.

### Bound quality reported beside the bounds
- `ari.bound_quality` reports the strictly-positive lower-bound count, the
  finite and unbounded upper-bound counts, and quantiles of the finite
  interval widths, so a vacuous interval can be told from an informative one.
- Every rate names its denominator. `_of_all` covers every coordinate
  compared, `_of_changed` only those whose symbol changed, and a rate over
  changed coordinates is `null` when nothing changed rather than zero.
  `n_unbounded` stays a count. The width quantiles carry
  `finite_width_basis`, since restricting them to changed coordinates changes
  what they mean.
- Predeclared movement thresholds report counts definitely below, definitely
  above, and unresolved. A coordinate with an unbounded upper bound can never
  be definitely below. These are thresholds on the representation, not on
  functional harm.
- Generic codec bounds and norm-constrained bounds go in separate blocks, the
  latter carrying its declared assumption. The schema requires that
  assumption to be non-empty.
- Measured: at the canonical probe the bounds resolve almost nothing. Adjacent
  regions touch, so a positive lower bound requires a move of two or more
  regions — asserted as an exact identity. At `n_bins=2` under 1e-3 drift
  every detected change is a single-region move, so every lower bound is zero
  and a 0.01 threshold leaves every change unresolved. Positive lower bounds
  first appear at 5e-2 drift. This is a result for this probe and this
  intervention.
- `cost_record` reports reference-state bytes, report bytes and runtime
  separately, each with the input count and machine it was measured on.
- `spec/report-schema.json` gains optional `bound_quality`,
  `norm_bound_quality` and `thresholds` on the change profile, each with
  declared properties and closed to additions. HER and aggregate ARI keep
  their exact-code-equality semantics; `ari_version` stays `0.1` and earlier
  reports still validate.

### Matched-budget protocol v2
- `experiments/matched-budget-detectors/PROTOCOL.md` is reissued as v2.
  The committed pilot was scored under v1, retained as
  `PROTOCOL-v1.md` beside it. Section numbering is unchanged so existing citations resolve.
- **The deployment alarm policy is declared, because the definition moves the
  answer more than any experimental choice does.** Measured on the cached
  panel, a raw-vector SHA-256 false-alarms on about 0.45% of inputs under
  benign conditions but on 50% of deployments, against 0% for the canonical
  code either way. v2 defines a deployment alarm as the per-input rate
  exceeding a calibrated threshold, not as "any input changed", and reports
  both units for every method.
- **Sizing corrected.** v1 quoted a two-sided interval; for an upper bound on
  a false-alarm rate the one-sided `1 - 0.05^(1/N)` is the right convention.
  200 control episodes establish only 1.49%, not 1%; 299 are needed. The
  collection target is now 300 control and 300 intervention episodes.
- **The degenerate null is planned for rather than discovered.** The canonical
  code gives zero alarms on every benign condition, so there is no threshold
  to trade and a nominal 5% FPR is unachievable. v2 fixes three rules in
  advance: report achieved zero with the bound the episode count supports,
  separate methods on sensitivity at zero achieved false alarms, and report
  ties among alarm-free methods as ties rather than injecting noise.
- An **episode** is now defined — one complete collection of the frozen input
  set under one configuration, in its own process, with its own manifest — and
  inputs within an episode are explicitly not episodes.
- New §8 fixes multiple comparisons (Holm–Bonferroni over the primary family,
  uncorrected values reported alongside), episode-level paired testing, power
  and stopping rules with no interim analysis of detection performance, and
  exclusion rules that admit collection faults only.
- Two baselines the comparison was missing: **budget-matched block hashes**,
  so a hash is given the same storage and can localise too, and a **float16
  reference** as the larger-storage accuracy anchor.
- `kl_div` and `js_div` are renamed `kl_from_logits` and `js_from_logits`.
  They already applied a softmax internally, so they always took logits; the
  name now says so. Nothing at runtime tells a logit vector from an embedding
  — both are float arrays — so a guard cannot catch the misuse and the name
  has to. Result keys are unchanged.
- **New §4.1: every comparison stays inside one probe.** KL and JS exist only
  at the logit probe, so a table placing them beside embedding-probe ARI would
  compare two measurements of two different objects. Families for the multiple
  comparison correction are now per probe, corrected separately and never
  pooled.

### Extent, quantized movement and location, beside equality
- A 32-byte SHA-256 detects every change and costs a fraction of a code,
  so equality is not where a code earns its size. `ari.change_profile`
  reports what a hash cannot: the fraction of coordinates whose symbol
  changed, where they sit (counts per contiguous block, with the short
  final block carrying its own denominator), and how far they moved.
- Movement comes from `semq.regions`, which bounds the value change from
  the ordered quantization regions. Requires a SEMQ build exposing
  `Context.quant_regions`; where it is absent the block is omitted rather
  than approximated, because a bound that cannot be verified is worse
  than no bound.
- **Measured, not assumed: the canonical probe cannot bound roughly half
  of its own detected changes.** At `n_bins=2` the alphabet has four
  regions and two of them are outermost, so on p99-calibrated unit-norm
  embeddings 20% of coordinates sit in a region with no outer edge, and
  47% (dim 384) / 46% (dim 1024) of the coordinates that changed have an
  infinite upper bound on their movement. Lower bounds are always
  available. `displacement.n_unbounded` reports the share rather than
  substituting a region width that does not exist.
- `reference_bytes` and `capabilities` state the storage each detector
  needs and what it can answer, from construction rather than from a
  measurement: a hash detects any change and locates none; a code
  locates changes and bounds movement to its own resolution; raw fp32
  gives movement exactly. No advantage is claimed in advance.
- `spec/report-schema.json` gains an optional `change_profile` on the
  semq detector. No existing field changes meaning, so `ari_version`
  stays `0.1` and reports signed before 2026-09-10 still validate and
  are not re-signed.

### Byte disagreement is no longer reported as symbol disagreement
- SEMQ QUANT packs several coordinates into one byte and packs whether or
  not the context asks it to, so `(cur != ref).mean()` over two code
  buffers is a **byte** change rate. Four measurements reported it as a
  symbol or coordinate rate. At `n_bins=8` an isolated change reads twice
  the coordinate rate, at `n_bins=2` four times, and the factor falls
  towards one as changes get dense, so a published number cannot be
  converted afterwards.
- New `ari.code_metrics` reports each quantity against its own
  denominator: whole-code equality, changed coordinates (count, fraction
  and indices), bit Hamming (bits and fraction, over code bits only), and
  the byte rate under a name that says bytes. `chunked_code_diff` compares
  a vocabulary encoded in chunks without reading one chunk's padding as
  the next chunk's coordinates.
- Migrated: `decoding-reproducibility/baselines.py` (`SEMQ Hbar` becomes
  `SEMQ coord change`, with `SEMQ byte change (legacy)` beside it),
  `decoding-reproducibility/run_matrix.py`,
  `drift-rank-profile/run.py` and `regime-discrimination/run_matrix.py`
  (`semq_hbar` becomes `semq_coord_change` and
  `semq_byte_change_legacy`). Audited and unchanged:
  `ari/metrics.py`, whose `Hbar` was already a bit count, and
  `probe-verifiability/near_tie_sweep.py`, whose codes are
  one-index-per-subspace and already per-symbol.
- `spec/report-schema.json` gains optional `bits_per_coordinate`,
  `n_coordinates`, `n_bits` and `bit_hamming_rate` on the semq detector.
  No existing field changes meaning, so `ari_version` stays `0.1` and
  reports signed before 2026-09-10 still validate and are not re-signed.
- Stored result files keep the names they were written with.
  `docs/figures/make_figures.py` reads either name.

### `build_report` emits the multi-detector condition shape natively
- `ari.report.build_report` now writes each condition as
  `{"detectors": {"semq": {...}}}` — the shape the v0.1 schema has accepted
  since the multi-detector migration, and the one every stored submission
  already uses. **Breaking for external callers of the reference
  implementation on two counts:** the emitted condition shape changes (the
  flat pre-migration form is no longer produced), and `fingerprint.dim` is
  now required (it derives the semq detector's `bytes_of_reference_state`).
  Both report forms remain valid under `spec/report-schema.json` v0.1, so no
  spec version change: this is a builder-API change, not a schema change.

### First GPU-measured panel row — Salesforce/SFR-Embedding-2_R
- Measured on an 8-condition GPU session (L40S, fp32, TF32 off, deterministic
  algorithms, BLAS pinned, HF revision pinned); capture protocol committed at
  [`experiments/deployed-agent-panel/sfr_capture.py`](experiments/deployed-agent-panel/sfr_capture.py)
  and documented in the panel RESULTS. First row whose raw per-condition
  float matrices are retained.

## [0.1-preview] — 2026-07

Initial public-ready draft.

### Self-hosted comparable core complete — `conc` + `time` = 1.000
- Ran `conc` and `time` for all 8 self-hosted models ([`ari/tools/selfhosted_conc_time.py`](ari/tools/selfhosted_conc_time.py)),
  so **every model — API and self-hosted — now carries the full core `{proc, conc, time}`.**
  All 8 score **1.000** on both: bit-reproducible under concurrency and across time.
- Directly closes the multi-threaded-BLAS confound on ARI-Bench. *Caveat:* 8 concurrent encodes on
  a single **shared** model instance (not thread-safe in PyTorch) gave a rare, reproducible ~0.1%
  single-input flip on bge-large (BLAS reduction-order under contention) — a serving-implementation
  artifact, not a reproducibility property (batched/replica serving and time are bit-exact). Recorded
  `conc = 1.000`; caveat documented. First direct sighting of BLAS-order code drift.

### Fingerprint registry — `κ` for the 9 `s_only` models
- Measured `(b, κ)` for the 9 remaining panel models via a new tool,
  [`ari/tools/drift_sweep.py`](ari/tools/drift_sweep.py) (source-agnostic: SentenceTransformer or
  black-box API). **7 fit cleanly** — voyage 2.489, cohere 2.657, mistral 1.484, gemini 2.547,
  nomic 2.924, mxbai 2.546, arctic 3.058 — so **11 of the 13 panel models now carry a full
  `(s, b, κ)`**. Registry range widens to `κ ∈ [1.48, 3.06]`, `b = 0.979 ± 0.028`.
- **2 models are `floor_limited`** (mpnet, MiniLM): `κ` cannot be cleanly fit due to a hard
  quantizer-boundary floor (~1% of code bytes flip under sub-ULP, σ-independent perturbation).
  Investigation **ruled out** dimension (nomic 768-d is clean), near-zero mass, and value
  discreteness; corrected the spec's earlier imprecise "d ≤ 768 shows a floor" caveat to the
  precise model-specific mechanism. **Their ARI is unaffected (bit-reproducible).**
- Validated the tool against the published fingerprint registry (same code-position Hamming as
  the reference sweep): bge-large 1.6%, bge-m3 1.1%, e5-large 7.2% (multilingual/corpus).

### Spec
- ARI-Canonical-v0.1 **frozen**: canonical probe (SEMQ QUANT, then named QBIN, n=2, 99th-pct calibration),
  reference model (bge-large-en-v1.5, s=0.0775 on ARI-Bench), + design rationale.
- ARI-Bench-v0.1: **frozen** input set — 1,000 real BEIR items (NFCorpus / SciFact / FiQA,
  512-char), hash-pinned.
- **Fingerprint registry** (`fingerprints-v0.1.csv`): published `(s, b, κ)` per model — full
  `(s, b, κ)` for 11 of the 13 panel models (+2 sweep-only decoders), `s` for all 13; 2 models
  (mpnet, MiniLM) are `floor_limited`.
- Condition set + report JSON schema (per-condition audit digests, `agent_class`,
  `input_content_hash`). **Aggregation:** headline ARI is the mean HER over the *comparable
  core* `{proc, conc, time}` (measurable for any agent class); `mach`/`prec`/`lib` are
  reported as self-hosted diagnostics, not folded into the headline (keeps API vs self-hosted
  apples-to-apples).

### Evidence (instrument)
- Drift-sensitivity benchmark: universal power law `b = 0.979 ± 0.028` (13-model registry), the
  `(s, b, κ)` fingerprint, instrument-class result. Probe-purity and cross-process experiments.
- **GPU determinism** (A10G + T4): an instrument validation on a *known* phenomenon (not a
  discovery). An isolation 2×2 attributes the cross-process drift to **TF32**: proc HER is
  1.000 with TF32 off and 0.65–0.88 with TF32 on, with the determinism flag having no effect —
  and TF32 is off by default in PyTorch 2.3, so a vanilla deployment is reproducible. TF32 also
  breaks GPU↔CPU equivalence (0.54–0.88); cross-GPU is portable; the gap is CPU↔GPU (mxbai).
  Reconciles with prior work (ReproRAG, arXiv 2509.18869). All invisible to cosine.

### Leaderboard
- **Deployed-Agent Panel**: 13 real, verified rows (5 embedding APIs + 8 self-hosted open
  models) on the frozen ARI-Bench-v0.1, each backed by a signed report. Proc-based ARI with
  95% bootstrap CIs.
- Gemini caching probe (reproducibility is genuine, not a common-string cache).
- **Precision cliff** (self-hosted `prec` diagnostic): all 8 open models are 1.000 on the core
  but the `prec` axis (bf16 vs fp32) collapses HER to ≈ 0 (mean 0.02) — proof the instrument
  catches self-hosted drift the moment a real axis is exercised, and that self-hosted
  reproducibility is config-dependent, not automatic.

### Tooling
- Reference harness (`ari/`): probe binding, agents, metrics, report builder, scorer.
- `run_report`, `build_ari_bench`, `capture_time_baseline`, pilots.

### Deployed-agent panel — API comparable core complete (`proc` + `conc` + `time`)
- The API headline ARI now averages the full core `{proc, conc, time}`, each with its own audit
  digest, with `time` at the **canonical ~76h (3-day)** gap: gemini 1.000, openai 0.845,
  mistral **0.699**, voyage **0.500**, cohere 0.169.
- Two axes each expose a provider-specific vulnerability no other surfaces:
  **`conc`** — Voyage collapses from proc 0.615 to conc **0.209** under a 64-way burst (requests
  routed to disagreeing backends) → ARI 0.500; **`time`** — Mistral drifts to **0.554** over 3
  days (vs 0.753 same-session), a silent multi-day change the 19h gap missed. Gemini holds 1.000
  on every axis. Mistral's `conc` measured at burst=16 (429 rate-limit at 64).
- Caveats: rates are input-distribution dependent.
