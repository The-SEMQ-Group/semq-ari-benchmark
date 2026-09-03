# ARI canonical condition set

*The condition set defines which environmental deltas the aggregate ARI averages over.*

An **environmental condition** is a tuple
`E = (process, machine, precision, library version, concurrency, time, batch shape)`. The
baseline run `E₀` fixes all of them; each canonical condition perturbs exactly one axis so
that drift is *attributable*.

| id | description | isolates | expected on a reproducible agent |
| --- | --- | --- | --- |
| `same` | same context, repeated immediately | probe purity + agent within-session determinism | deterministic agent: HER = 1.000; black-box API: a measurement (may be < 1.0) |
| `proc` | same machine, new process | BLAS thread schedule | HER < 1.0 iff multi-threaded BLAS non-determinism |
| `mach` | same CPU model, different machine | hardware drift | HER < 1.0 iff hardware-level float divergence |
| `prec` | same machine, fp16 vs fp32 vs bf16 | precision-induced drift | typically the largest HER drop |
| `lib` | same machine, library patch-version delta | version drift | HER < 1.0 iff a kernel changed |
| `conc` | same process, under load (other workers) | concurrency effects | usually near 1.0 |
| `time` | same machine, wall-clock gap > 24h | combined real-world | the cross-process replay failure mode (sleep/wake) |
| `batch` | same process, same inputs, different batch size | batch-shape sensitivity | HER < 1.0 iff padding or a shape-dependent kernel changes the result |

## `same`: a control for deterministic agents, a measurement for APIs

For a **deterministic agent** (self-hosted, pinned build), `same` is the probe-purity
control and must read HER = 1.000: re-encoding an identical input yields an identical code.
A value < 1.0 there means the *probe* is non-deterministic — exactly what disqualifies
stochastic quantizers (LSH / PQ / OPQ read 0.5 even on `same`). SEMQ passes it by
construction (the discrete-attractor property).

For a **black-box API**, we do not control the agent, and the agent may itself be
non-deterministic. There, `same` HER < 1.0 does **not** indicate probe impurity — the
probe's purity is held by the discrete-attractor property, checked once on
deterministic inputs and not re-litigated per report (a statement about this fitted,
frozen probe — [probe-validation](../docs/analysis/probe-validation.md)). Instead, `same` becomes a
**first-class measurement**: the agent's within-session determinism. Reports carry an
`agent_class` (`self_hosted` | `api`); the scorer enforces `same = 1.0` only for
`self_hosted`.

**Pilot finding (preliminary — one provider, one snapshot).** OpenAI
`text-embedding-3-large` is intermittently non-deterministic *per call*: a fraction of
identical repeated queries return a materially different embedding — verified at the float
level (~2,300 / 3,072 dimensions differ, magnitude ~1e-4; not last-bit noise) — which SEMQ
resolves as a code change. And `proc` (fresh client/connection) ≈ `same` (persistent
connection), so the non-determinism is **per-call, not connection-level**: for such an API
`proc` collapses to repeated-call stability and the fresh-context machinery is moot. This is
the drift ARI exists to expose, and no provider discloses it.

## Aggregation

**Headline ARI averages over the comparable core `{proc, conc, time}`** — the conditions
measurable for *any* agent class. `ARI(A) = mean over the present core conditions of HER(Eₖ)`.

This keeps self-hosted and black-box-API scores apples-to-apples. An API cannot observe
`mach` / `prec` / `lib` (they are provider-internal), so folding those into the average would
make a self-hosted agent — which honestly *exposes* those axes — score below an API that
merely hides them. `same` is reported but excluded from the average (deterministic agent:
definitionally 1.0; API: a diagnostic of within-session determinism).

**Diagnostics: `mach` / `prec` / `lib` / `batch`.** Measurable for a self-hosted agent, these are
reported per-condition but **not** folded into the headline ARI. They are where much of the
interesting self-hosted drift lives — in particular `prec` (fp16/bf16 vs fp32) is a
reproducibility *cliff*: across widely-used encoders, HER collapses from 1.000 (`proc`) to
≈ 0 under a precision change, entirely invisible to cosine / retrieval. Report them; read
them **alongside** the headline, not inside it.

`batch` is the odd one out: it is not an environment change at all, but a change in how the
caller groups its own inputs. It is a diagnostic rather than a core condition because a
provider that offers no batched endpoint cannot be measured on it, and because a caller who
fixes its batch size never sees it. Where it does move, it usually moves for the same reason
`prec` does — a shape-dependent kernel or a padding path. Report it when the agent exposes a
batch API; leave it absent otherwise. An absent `batch` and a `batch` of 1.000 are different
claims, and the leaderboard renders them differently.

Each condition also reports mean Hamming drift `H̄` and a drift spectrum, so drift is always
attributable to *which* axis caused it. An API's headline (core only) is a **measured lower
bound** on its true drift: any `mach` / `prec` / `lib` drift it has is unobservable from
outside, so a self-hosted and an API ARI are not strictly comparable even on the shared core.

## Reporting requirement

Because ARI measures a **system** property, every report must document the concrete
environment for `E₀` and each `Eₖ`: BLAS implementation and thread count, hardware
architecture, precision, and library versions. An ARI number without its environment
manifest is not third-party-verifiable.
