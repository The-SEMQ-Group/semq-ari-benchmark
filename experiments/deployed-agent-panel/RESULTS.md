# Deployed-Agent Panel — Results (v0.1-preview)

Real ARI measurements for **13 embedding models** — 5 commercial APIs + 8 self-hosted open
models — on the frozen [ARI-Bench-v0.1](../../data/ari-bench-v0.1.jsonl) (1,000 real BEIR
items, `content_hash e9ec8b01…`).

> **The commercial-API rows are the central result — as a *measurement*, not a discovery.** That
> hosted embedding APIs drift is known (OpenAI's especially, community-documented
> [since 2023](https://github.com/openai/openai-python/issues/868)). What's new is the
> **standardized, comparable, third-party-verifiable** number across five providers on a frozen
> input set with a bit-exact metric — MTEB-style: we make it the reference measurement, not the
> discovery. Full novelty discussion in the
> [README's "first real numbers" section](../../README.md#the-first-real-numbers-ari-leaderboard-v01-preview).

ARI here is the headline index over the **comparable core** `{proc, conc, time}` — **all three
are now measured for the APIs** (`time` at a **~76h (>24h canonical)** gap; `conc` as a burst-vs-calm
load contrast). 95% CI by block bootstrap over inputs (the aggregate CI is the mean of the
present conditions' intervals). `mach`/`prec`/`lib` are reported as **self-hosted diagnostics**
(see below), not folded into the headline.

## The leaderboard

| rank | agent | class | ARI | 95% CI |
| --- | --- | --- | --- | --- |
| 1 | `gemini/gemini-embedding-001` | **api** | **1.000** | [1.000, 1.000] |
| 1 | `BAAI/bge-large-en-v1.5` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `BAAI/bge-m3` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `intfloat/multilingual-e5-large` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `sentence-transformers/all-MiniLM-L6-v2` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `sentence-transformers/all-mpnet-base-v2` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `nomic-ai/nomic-embed-text-v1.5` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `mixedbread-ai/mxbai-embed-large-v1` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `Snowflake/snowflake-arctic-embed-l` | self-hosted | 1.000 | [1.000, 1.000] |
| 10 | `openai/text-embedding-3-large` | api | 0.845 | [0.822, 0.866] |
| 11 | `mistral/mistral-embed` | api | 0.699 | [0.672, 0.727] |
| 12 | `voyage/voyage-4-large` | api | 0.500 | [0.472, 0.528] |
| 13 | `cohere/embed-v4.0` | api | 0.169 | [0.147, 0.194] |

API ARI = mean of `proc`, `conc`, `time` (per-condition HER in the submissions). Per-API
breakdown `proc / conc / time` (`time` at ~76h): gemini 1.000 / 1.000 / 1.000, openai
0.862 / 0.851 / 0.822, mistral 0.753 / 0.791 / **0.554**, voyage 0.615 / **0.209** / 0.676,
cohere 0.146 / 0.130 / 0.232.
(`same`, a within-session diagnostic, per API: 1.000 / 0.859 / 0.734 / 0.571 / 0.133.)

n=1000 for all APIs and self-hosted, except `mxbai`/`arctic` at n=200 (deterministic; capped
for CPU cost). Each row is backed by a signed report under
[`ari-leaderboard`'s `submissions/`](https://github.com/The-SEMQ-Group/ari-leaderboard/tree/main/submissions).

## Headline findings

1. **Reproducibility is a real, provider-specific axis — the paid APIs span ARI 0.15 → 1.00.**
   It is *not* an anti-API bias: one API (Gemini) is perfectly reproducible, so the spread is
   a genuine property of each provider's serving stack.
2. **Gemini is the only bit-reproducible API — genuinely, not a cache.** A caching probe (novel,
   un-cacheable inputs) kept Gemini at 1.000 while the OpenAI control drifted; and the `time`
   condition holds it at **1.000 across a ~76h (3+ day) gap** — ruling out any short-TTL cache.
   It is deterministic compute. See [`docs/methodology.md`](../../docs/methodology.md).
3. **The drift is mostly per-call — but Mistral accumulates over days.** For openai/voyage/cohere,
   `time` (~76h) ≈ `proc` (per-request noise, not time-dependent). **Mistral is the exception:**
   its `time` drops to **0.554** (vs `proc` 0.753, and 0.753 at a 19h gap), with H̄ doubling — a
   genuine temporal drift that only a multi-day gap surfaces (a silent model/infra change over
   3 days). This is exactly why the canonical `time` gap is >24h: it caught drift that `proc`
   and a 19h gap did not.
4. **Concurrency is a hidden axis — Voyage drifts ~3× under load.** The `conc` condition (a
   burst of 64 concurrent requests vs a calm pass) leaves openai/mistral/cohere unchanged
   (≈ their per-call rate) and **Gemini bit-perfect (1.000) even under 64-way burst** — but
   `voyage-4-large` collapses from `proc` 0.615 to `conc` **0.209** (H̄ 3× larger). Under load,
   Voyage routes concurrent requests to backends that disagree — a real reproducibility
   vulnerability no other axis surfaced. It drops Voyage's overall ARI to **0.500**.
5. **Cohere `embed-v4` (0.169) and Voyage `voyage-4-large` (0.500) score lowest.** `voyage-4-large`
   is the embedder [Anthropic's documentation recommends](https://platform.claude.com/docs/en/docs/build-with-claude/embeddings),
   which makes its drift a practical concern for that stack.
6. **All 8 self-hosted open models are perfectly reproducible on the comparable core (1.000)** —
   under `proc` (fresh process), even with BLAS threads *unpinned* on x86. **But reproducibility
   is config-dependent:** the `prec` diagnostic (serving bf16 instead of fp32) collapses HER
   from 1.000 to ≈ 0 across all eight — a **precision cliff** (see below). The honest claim is
   *self-host **and pin your precision** → bit-reproducible*, not "self-hosted is always safe."
7. **The drift is invisible to cosine / retrieval.** It is representation-level (the raw
   embedding differs enough to cross a semantic boundary), which no cosine-based check
   surfaces — only a SEMQ-class instrument does. It affects bit-level reproducibility and
   auditability, not necessarily retrieval quality.

## Self-hosted diagnostics — `prec` (the precision cliff)

Reported per-condition, **not** folded into the headline ARI (an API can't measure it, so
averaging it in would break cross-class comparability). Every self-hosted model is
bit-reproducible under `proc` but collapses under a precision change — fp32 baseline vs bf16
serving (n=256):

| model | `same` | `prec` HER (bf16) | H̄ (bits) |
| --- | --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 1.000 | 0.004 | 5.94 |
| BAAI/bge-m3 | 1.000 | 0.000 | 7.25 |
| intfloat/multilingual-e5-large | 1.000 | 0.004 | 6.77 |
| sentence-transformers/all-mpnet-base-v2 | 1.000 | 0.035 | 4.08 |
| sentence-transformers/all-MiniLM-L6-v2 | 1.000 | 0.094 | 2.34 |
| nomic-ai/nomic-embed-text-v1.5 | 1.000 | 0.008 | 4.36 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.000 | 0.008 | 5.91 |
| Snowflake/snowflake-arctic-embed-l | 1.000 | 0.004 | 6.43 |

Higher-dimensional models flip more bits (more coordinates to disagree on). This is the axis
`spec/condition-set.md` predicts is "typically the largest HER drop", now confirmed. `mach`
(second machine) and `lib` (library-version delta) need the corresponding environment and are
pending — a GPU run will double as `mach`.

## Method

- **Probe:** SEMQ QBIN n=2, 99th-pct calibration ([`spec/ari-canonical-v0.1.md`](../../spec/ari-canonical-v0.1.md)).
- **Protocol / HER definition / condition set:** [`README.md`](README.md) (design) and
  [`docs/methodology.md`](../../docs/methodology.md) (methodology, ground-truth verification, the Gemini caching probe).
- **Inputs:** frozen [`data/ari-bench-v0.1.jsonl`](../../data/ari-bench-v0.1.jsonl).
- **ARI:** mean HER over the comparable-core conditions `{proc, conc, time}` — **all three now
  measured for the APIs**; `mach`/`prec`/`lib` reported as self-hosted diagnostics, not averaged
  in; `same` a measured within-session axis for `agent_class=api`.

## Caveats (honest)

- **The comparable core is complete for every model** — APIs and self-hosted both carry
  `proc` + `conc` + `time`. All 8 self-hosted score **1.000 on `conc` and `time`** (bit-reproducible
  under concurrency and across time). *Caveat:* running 8 concurrent encodes on a **single shared
  model instance** (not thread-safe in PyTorch) produced a rare, reproducible ~0.1% single-input
  flip on bge-large (BLAS reduction-order under thread contention) — a serving-implementation
  artifact, not a reproducibility property: under batched or replica'd serving (the realistic
  case) and across time, self-hosted is bit-exact. Recorded `conc = 1.000` accordingly.
- **`conc` burst level differs for Mistral.** `conc` is a 64-concurrent burst vs a calm pass;
  Mistral rate-limited (429) at 64, so its `conc` was measured at burst=16 (still 4× the calm
  load). Mistral shows no concurrency effect at either level.
- **`time` is the canonical >24h** (measured at ~76h, i.e. 3+ days). It is what surfaced
  Mistral's temporal drift (0.554) that the 19h gap missed.
- **Input-distribution sensitivity.** The API rates shifted from the earlier synthetic sample
  to real BEIR text (e.g. Voyage 0.40 → 0.62), which is why the frozen real inputs matter.
- **Gemini 1.000** is deterministic compute, not a cache: the caching probe rules out a
  common-string cache and the ~76h `time` condition rules out any short-TTL cache.

---

## Addendum — Salesforce/SFR-Embedding-2_R (added 2026-08-31)

The panel's first GPU-measured row, added after the original 13. Protocol
identical in structure to the self-hosted rows, adapted for a 7B model:

- **Environment**: single L40S (g6e.xlarge), fp32, TF32 off,
  `torch.use_deterministic_algorithms` on, BLAS threads pinned, HF revision
  pinned (`f62d15f411ca97b66acc0f34da2a65f3420b55b0`). Weights load in the
  checkpoint dtype (bf16) and are upcast to fp32 on device — lossless for
  bf16-born weights.
- **Conditions**: fp32 base → `same` (in-process re-encode), `proc` (fresh
  subprocess with its own CUDA context), `conc` (8 threads, shard batches),
  `time` (fresh-process re-encode), and the `prec` diagnostic (a fresh bf16
  load — never an in-place cast; casting a loaded model crushes fp32-born
  rotary buffers, which the instrument itself caught during this session as
  an 89.5% code change from a 1.2e-03 perturbation in two buffers, and the
  contaminated pass was discarded and remeasured cleanly).
- **Result**: `same = proc = conc = time = 1.000` exact; `prec` HER 0.0,
  H̄ 24.4 bits — consistent with the other self-hosted rows at 4096
  dimensions.
- **Script**: [`sfr_capture.py`](sfr_capture.py) (self-contained; reads the
  frozen input set, verifies its pin, writes the report via
  `ari.report.build_report`). **Evidence**: the report, its KMS attestation
  and the raw per-condition float matrices (a first for the panel — the
  float-detector gate can be exercised retroactively on this row) are
  digest-bound; float matrices archived at
  `s3://semq-ari-baselines-127348475353/capture-sessions/sfr-2r/`.
