# Panel methodology & verification

**How the panel numbers are produced, and why they're trustworthy.** The deployed-agent panel's full results (13 agents) live in [`../RESULTS.md`](../experiments/deployed-agent-panel/RESULTS.md); this document is the *methodology* — the
HER definition, the ground-truth float-level verification, and the adversarial probes — using
OpenAI as the worked example plus the Gemini caching probe. Headline ARI averages `proc` and
`time` (~76h canonical) and `conc` for the APIs.

## What HER measures here (read this before quoting the number)

HER is the fraction of inputs whose **SEMQ code of the embedding** is bit-identical between
two calls. It measures whether OpenAI's *raw embedding vector* for an identical input landed
in the same SEMQ bins — **not** whether any downstream answer or completion changed. The
precise claim is at the representation level:

> In **~15% of identical repeated calls** (95% CI 13–17%, n=1000) to the
> `text-embedding-3-large` endpoint, OpenAI's raw embedding differs enough to cross a semantic
> boundary that SEMQ resolves as a code change — a divergence **invisible to cosine similarity
> / retrieval** and visible only to a SEMQ-class instrument.

This is *not* "OpenAI's answers change ~15% of the time." It is an embedding-level,
representation-reproducibility statement. **Scope:** it affects **bit-level reproducibility
and auditability** (the same query is not guaranteed to yield the same stored/compared
representation), not necessarily retrieval quality (at this magnitude the cosine top-K rarely
moves — which is exactly why no cosine-based check would ever surface it).

## Setup

- Agent: `openai/text-embedding-3-large` (dimensions=3072, encoding_format=float, one input
  per request, pinned model id).
- Probe: real SEMQ QBIN n=2 (`semq==1.2.0`), 99th-pct calibration.
- Run: **n=1000** with a bootstrap CI over inputs (illustrative; the canonical published values are on the frozen ARI-Bench in RESULTS.md).
  `same` = repeats on a persistent client; `proc` = repeats on a fresh client/connection.
- Data: [`openai_proc_pilot_n1000.json`](../experiments/deployed-agent-panel/artifacts/openai_proc_pilot_n1000.json),
  [`openai_proc_pilot_n1000.json`](../experiments/deployed-agent-panel/artifacts/openai_proc_pilot_n1000.json).

## Result (n=1000, K=3; 95% CI by block bootstrap over inputs)

| condition | HER | 95% CI | reading |
| --- | --- | --- | --- |
| `same` | **0.852** | [0.834, 0.870] | within-session representation determinism — ~15% of identical repeated calls diverge |
| `proc` | **0.851** | [0.833, 0.868] | fresh-context ≈ same (CIs overlap) |

So **~15% (roughly 1 in 7)** identical repeated calls to `text-embedding-3-large` produce a
different embedding code (95% CI 13–17%). The earlier small-n runs read ~19% (n=300); the
rate refined downward with n=1000 + a proper bootstrap CI — which is why the CI, not a bare
point estimate, is what we publish. When a code differs it moves by a small number of bits —
the small-magnitude, high-pass regime SEMQ is built for.

> These numbers are from an early illustrative run on a dry-run sample, kept here because the
> *reasoning* is what this document teaches. The **canonical published** OpenAI values, on the
> frozen ARI-Bench-v0.1, are `same` = 0.859, `proc` = 0.862, `conc` = 0.851, `time` (~76h) = 0.822, and the
> headline **ARI = 0.845** [0.822, 0.866] — the story is identical. See
> [`RESULTS.md`](../experiments/deployed-agent-panel/RESULTS.md).

## Why `proc` ≈ `same` (and why `proc > same` is not a finding)

At n=300, `proc` (0.823) came out marginally above `same` (0.806). This is **not** "less drift
across processes than within a session." It is:

1. **Not statistically significant** — at n=1000 the 95% CIs fully overlap (`same`
   [0.834, 0.870] vs `proc` [0.833, 0.868]); the n=300 gap was sampling noise.
2. **Mechanistically expected** — for a remote API we do not control the provider's routing. A
   fresh client and a persistent one both land on OpenAI's replica pool independently, so
   `same` and `proc` measure the *same thing* for a black-box API (repeated calls hitting a
   rotating backend). Their tiny difference is sampling noise.

This reinforces the real conclusion: the drift is **per-call, not connection-level**. For an
API the whole condition set collapses (we do not control the server), which is exactly why an
API's ARI is a **black-box drift number, not an attributable one**.

## Self-hosted contrast (bge-large-en-v1.5) — across hardware and thread settings

The same SEMQ probe against the reference model we control, n=64. `same` = re-encode in the
same process; `proc` = re-encode in a **fresh subprocess** (the real cross-process test).
Measured on three configs:

| config | BLAS / threads | `same` HER | `proc` HER |
| --- | --- | --- | --- |
| Apple Silicon | Accelerate, multi-thread | 1.000 | 1.000 |
| x86 (c7i), **unpinned** | OpenBLAS 0.3.27, torch 4-thread | **1.000** | **1.000** |
| x86 (c7i), pinned | threads = 1 | 1.000 | 1.000 |

**Self-hosted bge-large is bit-reproducible cross-process on every config — multi-threaded,
pinned or not, on two BLAS stacks.** Zero Hamming everywhere. Data:
[`selfhosted_bge_large_x86_unpinned.json`](../experiments/deployed-agent-panel/artifacts/selfhosted_bge_large_x86_unpinned.json),
[`..._x86_pinned.json`](../experiments/deployed-agent-panel/artifacts/selfhosted_bge_large_x86_pinned.json),
[`selfhosted_bge_large_pinned.json`](../experiments/deployed-agent-panel/artifacts/selfhosted_bge_large_pinned.json) (Apple Silicon).

| agent | `same` HER | `proc` HER | reading |
| --- | --- | --- | --- |
| `openai/text-embedding-3-large` (API) | 0.852 [0.834, 0.870] | 0.851 | ~15% per-call representation drift |
| `BAAI/bge-large-en-v1.5` (self-hosted) | **1.000** | **1.000** | bit-identical, zero drift |

**Governance headline:** self-host the reference model → **bit-reproducible** (verified on
Apple Silicon *and* x86, multi-threaded, with and without pinned threads). Use the API → ~15%
(95% CI 13–17%) of identical embedding calls silently diverge at the representation level —
undisclosed, and invisible to any cosine-based check.

**On BLAS non-determinism — closing the multi-threaded-BLAS confound, honestly.** We tested the exact
concern — self-hosted *unpinned* on x86 with multi-threaded OpenBLAS — and it was **still
bit-reproducible** (`proc` = 1.000). So the comparison is not an artifact of thread pinning;
self-hosted reproducibility is **robust**, and we make no "pinned threads" caveat.
Conversely, we did **not** reproduce cross-process BLAS drift in this controlled run — modern
torch/OpenBLAS matmul was deterministic here. The historical cross-process-replay incident
likely involved a hardware change across a sleep/wake cycle, not pure thread scheduling. We
therefore **do not claim "unpinned self-hosting drifts."** We claim self-hosting is robustly
reproducible and SEMQ is the instrument that *verifies* it (and would catch drift if a stack
introduced it). Reproducing a live *cross-process* BLAS-drift case is a separate, still-open
follow-up. (The `conc` condition later did surface a **concurrency** BLAS effect: 8 threads
encoding on a single shared model instance produced a rare, reproducible ~0.1% single-input
flip on bge-large — the reduction order shifting under thread contention. It is a shared-instance
serving artifact, not a batched-serving property, but it is the first direct sighting of
BLAS-order code drift, and SEMQ resolved it.)

## Ground-truth verification (why this is real, not a pipeline bug)

`same` HER < 1.0 contradicted the assumption that `same` is a 1.0 control, so it was verified
at the **float level** before any conclusion:

- 4 rapid repeats of one identical text → **bit-identical** vectors (a short-window cache).
- 30 texts embedded in two *separated* passes (each text's two calls ~30 apart, mirroring the
  pilot) → **3 of 30 texts differed at the float level**, and when they differed, **~2,300 /
  3,072 dimensions** changed, magnitude **~1e-4** — a materially different vector from a
  different backend/replica, not last-bit noise.

Conclusion: `text-embedding-3-large` is intermittently non-deterministic per call at the
embedding level. The harness is correct; the differing codes trace to genuinely different
embeddings.

## Gemini reproducibility — determinism, or caching?

`gemini-embedding-001` is the only API measured at ARI 1.000, which invites the obvious
objection: *"that's server-side caching of repeated strings, not deterministic compute."*
Tested it directly:

- Built 150 **novel** inputs — a unique nonce appended to each real text, so Gemini has
  certainly never seen (and cannot have cached) them.
- **Gemini on novel inputs: ARI 1.000** — reproducible even on guaranteed-uncached strings.
- **OpenAI on the *same* novel inputs (control): ARI 0.807** — still drifts, proving the
  probe detects non-determinism when it exists.

Conclusion: Gemini's reproducibility is **not** an artifact of a common-string cache — it
holds where a lookup cache cannot help, while the control drifts on those very inputs.
Whether Gemini achieves it by deterministic compute or a deliberate short-TTL result cache,
the reproducibility is **real and Gemini-specific**. The short-TTL-cache-vs-determinism
distinction is now **settled by the `time` condition**: Gemini holds **1.000 across a ~76h
(3-day) gap**, far beyond any plausible result-cache TTL — deterministic compute.

## What this changed (spec + scorer)

- ARI now treats `same` as a **measured axis** for `agent_class=api` (within-session
  representation determinism, may be < 1.0), not a 1.0 control — probe purity is guaranteed by
  the discrete-attractor property, proven once, not re-checked on a non-deterministic agent.
  See [`../spec/condition-set.md`](../spec/condition-set.md).
- For a per-call-non-deterministic API, `proc` collapses to repeated-call stability.

## Before publishing

- [x] Larger n with a bootstrap CI over inputs (n=1000) — the defensible rate.
- [x] x86 unpinned self-hosted run — still bit-reproducible (`proc`=1.0), so the confound is
      closed *favourably* (no pinning caveat needed).
- [x] Temporal replication — the `time` condition is measured at the canonical ~76h gap; the
      drift is per-call (time ≈ proc), except Mistral, which accumulates over days (0.554).
- [x] Voyage + Cohere — measured; both are live leaderboard rows (0.500 / 0.169). Gemini
      provides the deterministic contrast (1.000).
- [~] A live **cross-process** BLAS-drift case is still not reproduced on modern x86/Apple; the
      `conc` condition did surface a **concurrency** BLAS effect (a rare ~0.1% shared-instance
      flip), so the "SEMQ catches BLAS drift" claim now has one direct sighting, still caveated.
