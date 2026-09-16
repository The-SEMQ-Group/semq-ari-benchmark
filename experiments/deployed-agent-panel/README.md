# Deployed-Agent Panel

**Status: RUN (v0.1-preview).** Results for 13 agents are in [`RESULTS.md`](RESULTS.md);
methodology + verification in [`docs/methodology.md`](../../docs/methodology.md). This document is the **design/protocol**.

This is the experiment that produces the first real ARI numbers for the public leaderboard.
The drift-sensitivity benchmark proved the *instrument* works; this measures *deployed agents*
with it.

## Objective

Produce a signed [ARI report](../../spec/report-schema.json) for each agent in the panel,
under the canonical condition set ([`spec/condition-set.md`](../../spec/condition-set.md)),
and populate the board's `leaderboard.json` (now in the
[ari-leaderboard](https://github.com/The-SEMQ-Group/ari-leaderboard) repo) with
hash-verifiable rows.

Success = each agent has an `ARI ∈ [0,1]` with per-condition HER/H̄ and 95% CIs, plus an
audit trail a third party can re-verify.

## Panel (first tranche)

Two classes, because they exercise the condition set very differently.

### A. Embedding APIs — measuring *undisclosed provider drift*

`openai/text-embedding-3-large`, `voyage-4-large`, `cohere/embed-v4.0`,
`mistral/mistral-embed`, `gemini/gemini-embedding-001` (as run — see [`RESULTS.md`](RESULTS.md)).

We do **not** control the provider's environment — which is exactly the point. For an API,
ARI measures the drift the provider introduces *silently* behind a stable endpoint:
infrastructure swaps, model-snapshot updates, load-dependent routing. That is a number no
provider publishes and no buyer can otherwise see.

Applicable conditions: `same` (control), `proc` (fresh-context resampling — see below),
`conc` (burst vs idle), `time` (calls ≥24h apart). `mach` / `prec` / `lib` are
**provider-internal and unobservable** — any drift they cause surfaces inside `proc`/`time`
and is reported as such. API ARI is therefore averaged over `{proc, conc, time}` and flagged
as a **reduced-condition** score.

Because we cannot control the provider's process/machine, the API score is a **measured
lower bound on provider drift**, not a controlled measurement: we can only *induce* a
different backend probabilistically, so true internal non-determinism is ≥ what we report.
We pin everything client-side so any observed drift is attributable to the provider —
identical request bytes, all determinism-affecting params fixed (dimensions, encoding
format, truncation), **model snapshot pinned** where the provider exposes one (so `time`
drift is pure infrastructure drift, not model updates), same region endpoint, and one input
per request (no batch-composition confound). Each row is labelled `pinned` or `floating`
per its snapshot handling.

### B. Self-hosted on AWS — the full condition set

Eight widely-used open encoders — `BAAI/bge-large-en-v1.5`, `BAAI/bge-m3`,
`intfloat/multilingual-e5-large`, `sentence-transformers/all-mpnet-base-v2`,
`sentence-transformers/all-MiniLM-L6-v2`, `nomic-ai/nomic-embed-text-v1.5`,
`mixedbread-ai/mxbai-embed-large-v1`, `Snowflake/snowflake-arctic-embed-l` — served by us so
we control every axis (as run — see [`RESULTS.md`](RESULTS.md)).

Measurable conditions: the full `{proc, mach, prec, lib, conc, time}`. The **headline ARI**
still averages only the comparable core `{proc, conc, time}` (so self-hosted and API scores
stay apples-to-apples); `mach`/`prec`/`lib` are reported as **diagnostics**. This is where the
condition-by-condition decomposition lives — e.g. the measured `prec` cliff (bf16 vs fp32
drops HER from 1.000 to ≈ 0; see [`RESULTS.md`](RESULTS.md)).

## Condition set → concrete AWS realisation

| condition | how we realise it (self-hosted) | APIs |
| --- | --- | --- |
| `same` | re-encode the batch twice in one process | K calls back-to-back on one persistent connection |
| `proc` | restart the server process on the same instance, re-encode | fresh client + fresh connection, K resamples over a window (best-effort backend diversity; rotate key if provider routes by key) |
| `mach` | second instance of the **same type** (same GPU/CPU model, different physical host) | n/a (provider-internal) |
| `prec` | same instance, serve fp32 vs fp16 vs bf16 | n/a |
| `lib` | second env with a patched framework/kernel minor version | n/a |
| `conc` | encode under concurrent load vs idle | burst of parallel calls |
| `time` | re-encode after a ≥24h wall-clock gap | calls ≥24h apart |
| cross-region (extra) | second instance in a different region | (surfaces inside `time`) |

**Instance plan (real multi-instance):**

- 2× GPU instances of one family (e.g. `g5.xlarge`, A10G) in the same region → `mach`.
- 1× GPU instance of the same family in a **second region** → cross-region.
- 1× CPU instance (e.g. `c7i`) for the BLAS-thread realisation of `proc`/`mach` on the
  1024-d encoders (the σ ≈ 1e-6 regime the instrument is tuned for).
- A second software env (pinned patched framework versions) on one instance → `lib`.

GPU embedding is cheap once loaded; the panel is a few instance-hours. Rough budget: **a
few tens of USD** for the self-hosted side, single-digit USD for the APIs.

## Probe, inputs, procedure

- **Probe:** SEMQ QBIN, `n_bins=2`, 99th-pct calibration — the canonical probe
  ([`spec/ari-canonical-v0.1.md`](../../spec/ari-canonical-v0.1.md)). Calibrate once per
  model on the reference distribution; freeze `s`.
- **Inputs:** ARI-Bench v0.1 ([`spec/ari-bench-v0.1.md`](../../spec/ari-bench-v0.1.md)) — a
  fixed, hash-pinned N = 1,000 slice of BEIR (standard data, no authoring), with stable
  ordering so audit hashes are comparable.
- **Per (agent, condition):**
  1. Encode all inputs under the baseline `E₀`; store `hash(SEMQ_code(x))`.
  2. Re-encode under `Eₖ`; compute `HE = 1[code_0 = code_k]` and `H = Hamming` per input.
  3. Aggregate to `HER(Eₖ)`, `H̄(Eₖ)`, with 95% bootstrap CIs (1,000 resamples).
  4. `ARI = mean HER over the applicable averaged conditions`.
  5. Emit the report JSON with the environment manifest + SHA-256 audit hashes.
- **Determinism hygiene:** pin BLAS threads on the baseline so measured drift is attributed
  to the *condition under test*, not the host's own noise. Record the full environment.

## Statistical plan

- 1,000 inputs (frozen ARI-Bench-v0.1) × per-condition, 1,000 bootstrap resamples for HER CIs.
- Report both HER (bit-exact equality rate) and H̄ (mean Hamming) — HER is the headline,
  H̄ is the severity when it drifts.
- Pre-register the expectation: **at least one API scores ARI < 0.99** (undisclosed drift).
  Self-hosted with pinned everything should approach 1.0 on `proc`/`mach` and drop most on
  `prec`.

## Deliverables

1. One signed ARI report per agent under the board's `submissions/` (in `ari-leaderboard`).
2. Populated `leaderboard.json` (real rows, `verified: true`).
3. A short `RESULTS.md` here with the panel table and the per-condition decomposition.

## Open questions (resolve before running)

- ~~**ARI-Bench curation.**~~ **Resolved:** ARI-Bench v0.1 is a fixed, hash-pinned N = 1,000
  slice of BEIR (standard data — no prompt authoring). See
  [`spec/ari-bench-v0.1.md`](../../spec/ari-bench-v0.1.md). Remaining task is mechanical:
  fix the slice + ordering + content hash.
- ~~**Probe availability.**~~ **Resolved:** the probe is the `semq` SDK, `pip`-installable —
  from a private package index today, from public PyPI once the SEMQ SDK ships (a separate
  release track from this benchmark) — so it drops into the serving/build environments
  directly.
- **API `proc` semantics — pilot run, direction updated (see [methodology](../../docs/methodology.md)).** A first
  pilot against OpenAI `text-embedding-3-large` (n=50, K=5, real SEMQ probe) found that
  `proc` (fresh client/connection) ≈ `same` (persistent connection): the non-determinism is
  **per-call, not connection-level**. So the elaborate fresh-context machinery is moot for
  this provider — `proc` collapses to **repeated-call stability**, and even `same` < 1.0
  (the provider is intermittently non-deterministic per call; verified at the float level).
  Consequences: (1) `agent_class=api` reports treat `same` as a measured axis, not a 1.0
  control (spec + scorer updated); (2) one-per-request stays for v0.1 (no batch confound);
  (3) still open before freezing: confirm the pattern on Voyage / Cohere (they may be
  deterministic, which would make fresh-context matter for *them*) and quantify the rate at
  larger n. K over a window remains the knob for providers that *do* show connection-level
  drift.
- ~~**Snapshot pinning.**~~ **Decided:** pin the model snapshot wherever the provider exposes
  one (API ARI then measures pure infrastructure drift, not model updates); if a provider
  offers no pin, that column runs `floating`. Every row is labelled `pinned` / `floating`.
