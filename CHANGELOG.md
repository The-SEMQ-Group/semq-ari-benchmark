# Changelog

All notable changes to ARI (the spec, the benchmark, and the reference implementation).
This project follows [semantic-ish versioning](https://semver.org/) for the **spec** (v0.1,
v0.2, …); the leaderboard tracks a preview label until the spec is frozen.

## [Unreleased]

### Matched-budget protocol v2
- `experiments/matched-budget-detectors/PROTOCOL.md` is reissued as v2.
  The committed pilot was scored under v1, which git holds at commit
  `57d91e7`. Section numbering is unchanged so existing citations resolve.
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
- ARI-Canonical-v0.1 **frozen**: canonical probe (SEMQ QBIN n=2, 99th-pct calibration),
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
