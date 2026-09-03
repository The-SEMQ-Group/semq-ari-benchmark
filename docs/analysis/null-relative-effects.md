# Audit: condition-vs-`same` effects across the 13 submissions

**Scope.** This document is purely analytical. It does not modify `spec/` or the
board (the `leaderboard.json` and app now in the
[ari-leaderboard](https://github.com/The-SEMQ-Group/ari-leaderboard) repo),
recomputes no published `ARI`, and rewrites no submission. Every HER, `HER_ci` and
`ARI` figure below is a literal quote from the board's `submissions/*.json`.

## 0. The problem in one sentence

The published ARI is `mean(proc, conc, time)` — the mean over the comparable
*core* defined in [`spec/condition-set.md`](../../spec/condition-set.md).
`same` is measured and reported but sits outside that mean. That is correct
as a definition, but it produces a headline that conflates two different
questions:

- **How much measurement noise does the agent carry on its own?** — that is
  what `same` measures.
- **How much does each environmental condition move it?** — that is what
  `HER(condition) − HER(same)` should measure, not `HER(condition)` in
  isolation.

For a deterministic `self_hosted` agent, `same = 1.000` by construction, so
the 3-condition ARI is already relative to a known zero. For an `api` agent,
`same` is a *measurement* (it can be < 1.0), and the published ARI averages
absolute HER, not HER relative to `same`. If `same` itself is low and noisy,
an ARI that resembles `same` says nothing about the condition's effect — it
says the agent is that noisy even when nothing changes.

## 1. Method

For each agent and each present condition `k`:

```
delta(k) = HER(k) − HER(same)
overlap(k) = do the HER_ci(k) and HER_ci(same) intervals overlap
```

Classification (in this order, by explicit instruction of the task):

1. **Apparent improvement** — `HER(k) > HER(same)`, regardless of CI
   overlap. No environmental condition should *improve* reproducibility
   above the immediate-repeat control; a point estimate landing above it is
   a sign of measurement noise, not of a real condition effect. It is
   reported explicitly, not discarded.
2. **Real effect** — `HER(k) ≤ HER(same)` and the CIs do **not** overlap.
3. **Indistinguishable from noise** — `HER(k) ≤ HER(same)` and the CIs
   overlap.

This is a simple non-overlap test (conservative, no multiple-comparison
correction); it is not a formal hypothesis test, but it suffices to separate
"this is noise" from "this cannot be noise".

**Data availability.** All 13 submissions include `HER_ci` for every
condition they report — no CI is missing for the comparisons in this
document. What *is* missing, in all 13 submissions without exception, are
the `mach` and `lib` conditions: no submission reports them, even though
`spec/condition-set.md` invites `self_hosted` agents to do so as a
diagnostic. Nothing can be said about those two axes with the current data;
this is flagged rather than assumed to be near 1.0.

## 2. Full table: HER per condition, published ARI, effect vs. `same`

### 2a. `api` agents (5) — the full core {same, proc, conc, time}

| agent | same (CI) | proc (CI) → Δ, class | conc (CI) → Δ, class | time (CI) → Δ, class | published ARI |
|---|---|---|---|---|---|
| cohere/embed-v4.0 | 0.133 (0.111–0.154) | 0.146 (0.124–0.169) → +0.013, **apparent improvement** (CI overlaps) | 0.130 (0.110–0.152) → −0.003, **indistinguishable** | 0.232 (0.206–0.262) → +0.099, **apparent improvement** (CI does NOT overlap) | 0.169333 |
| openai/text-embedding-3-large | 0.859 (0.837–0.881) | 0.862 (0.840–0.883) → +0.003, **apparent improvement** (CI overlaps) | 0.851 (0.830–0.872) → −0.008, **indistinguishable** | 0.822 (0.797–0.844) → −0.037, **indistinguishable** | 0.845000 |
| mistral/mistral-embed | 0.734 (0.708–0.759) | 0.753 (0.727–0.778) → +0.019, **apparent improvement** (CI overlaps) | 0.791 (0.765–0.816) → +0.057, **apparent improvement** (CI does NOT overlap) | 0.554 (0.523–0.587) → −0.180, **real effect** | 0.699333 |
| voyage/voyage-4-large | 0.571 (0.541–0.600) | 0.615 (0.585–0.644) → +0.044, **apparent improvement** (CI overlaps) | 0.209 (0.185–0.233) → −0.362, **real effect** | 0.676 (0.645–0.706) → +0.105, **apparent improvement** (CI does NOT overlap) | 0.500000 |
| gemini/gemini-embedding-001 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) → 0.000, **indistinguishable** | 1.000 (1.000–1.000) → 0.000, **indistinguishable** | 1.000 (1.000–1.000) → 0.000, **indistinguishable** | 1.000000 |

### 2b. `self_hosted` agents (8) — core {same, proc, conc, time} + the `prec` diagnostic

For the 8 `self_hosted` agents, `same = proc = conc = time = 1.000` exactly
(CI `[1.0, 1.0]`) by construction — SEMQ's discrete-attractor property makes
these four columns trivial and information-free: the published ARI
(`mean(proc,conc,time)`) is 1.000 for all eight without exception, and none
can show a "real effect" in the core because there is no variance to
measure. The only column carrying signal is the `prec` diagnostic, reported
but excluded from the headline:

| agent | same/proc/conc/time | prec HER (CI) → Δ vs. same, class | published ARI |
|---|---|---|---|
| BAAI/bge-large-en-v1.5 | 1.000 (all) | 0.003906 (0.000–0.011719) → −0.996094, **real effect** | 1.000000 |
| BAAI/bge-m3 | 1.000 (all) | 0.000000 (0.000–0.000000) → −1.000000, **real effect** | 1.000000 |
| Snowflake/snowflake-arctic-embed-l | 1.000 (all) | 0.003906 (0.000–0.011719) → −0.996094, **real effect** | 1.000000 |
| intfloat/multilingual-e5-large | 1.000 (all) | 0.003906 (0.000–0.011719) → −0.996094, **real effect** | 1.000000 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.000 (all) | 0.007812 (0.000–0.019531) → −0.992188, **real effect** | 1.000000 |
| nomic-ai/nomic-embed-text-v1.5 | 1.000 (all) | 0.007812 (0.000–0.019531) → −0.992188, **real effect** | 1.000000 |
| sentence-transformers/all-mpnet-base-v2 | 1.000 (all) | 0.035156 (0.015625–0.058594) → −0.964844, **real effect** | 1.000000 |
| sentence-transformers/all-MiniLM-L6-v2 | 1.000 (all) | 0.093750 (0.058594–0.128906) → −0.906250, **real effect** | 1.000000 |

`mach` and `lib`: absent from all 8 self-hosted submissions. There is no
data to classify those axes.

## 3. The three specific cases to confirm

**Cohere — `same` 0.133, ARI 0.169: the score is essentially its own noise.
Confirmed, and more strongly than the preliminary reading suggested.** None
of the three core conditions shows a real degradation effect relative to
`same`: `conc` is indistinguishable (its CI overlaps `same`'s almost
entirely), and both `proc` and `time` land **above** `same` (apparent
improvement). `time` even does so with non-overlapping CI (0.206–0.262 vs.
`same`'s 0.111–0.154) — that is, there is a statistically distinguishable
difference, but in the direction that cannot be a real causal effect of the
condition (a >24 h gap cannot make the agent *more* reproducible than
repeating it immediately). That is evidence that Cohere's "noise" is not a
stable interval around one value: it fluctuates enough between runs for the
ordering `same < time` to be statistically solid and still mean nothing
about the effect of `time`. The 0.169 ARI is not measuring sensitivity to
conditions; it is measuring the same internal instability already visible
in `same`.

**OpenAI — `proc` 0.862 vs. `same` 0.859: indistinguishable. Confirmed, and
it extends to the other two.** `proc` (CI 0.840–0.883) overlaps `same` (CI
0.837–0.881) almost completely — the clearest overlap case in the dataset.
But the same holds for `conc` (0.830–0.872) and `time` (0.797–0.844): all
three core conditions are indistinguishable from `same` by CI. OpenAI's
published ARI (0.845) is not statistically distinct from its own `same`
(0.859) in any of its three components. The pilot finding
`spec/condition-set.md` already documents (per-call nondeterminism,
`proc` ≈ `same`) is exactly what shows up here: OpenAI's ARI does not
measure a condition effect, it measures the same per-call nondeterminism
already present in `same`.

**Voyage — `conc` 0.209 vs. `same` 0.571: the only strong condition effect
in the dataset. Confirmed.** `conc`'s CI is 0.185–0.233; `same`'s is
0.541–0.600 — zero overlap, and the gap (Δ = −0.362) is, by a margin, the
largest across the 5 `api` submissions and across all 13. It is the only
case in the whole dataset where a core condition degrades significantly
below `same` for an `api` agent. Note: Voyage also shows `time` as an
apparent improvement with non-overlapping CI (0.645–0.706 vs. `same`'s
0.541–0.600, Δ = +0.105) — as causally inexplicable as the Cohere case. So
Voyage's ARI (0.500) mixes one genuine, large condition effect (`conc`)
with a second component (`time`) that is, again, statistically
distinguishable measurement noise in the wrong direction. Averaging the
three hides that only one of them is real signal.

**An additional finding, not explicitly requested but visible in the same
table:** Mistral shows the same pattern as Voyage — `conc` as an apparent
improvement with non-overlapping CI (Δ = +0.057) and `time` with a real
degradation effect (Δ = −0.180, CI 0.523–0.587 vs. `same`'s 0.708–0.759).
Of the 5 `api` agents, only Mistral and Voyage carry a real condition
effect, and only Voyage carries it on the condition that intuitively
dominates its "worst case" (`conc` for Voyage, `time` for Mistral) while
the other strong condition is also an ambiguous apparent improvement.

## 4. The 8 self-hosted tied at 1.000: what `prec` says

The headline excludes `prec`, which is the only condition where the 8
differ. Values (from table 2b), ordered best to worst HER under `prec`:

| rank | agent | prec HER | prec CI |
|---|---|---|---|
| 1 | sentence-transformers/all-MiniLM-L6-v2 | 0.093750 | 0.058594–0.128906 |
| 2 | sentence-transformers/all-mpnet-base-v2 | 0.035156 | 0.015625–0.058594 |
| 3= | mixedbread-ai/mxbai-embed-large-v1 | 0.007812 | 0.000000–0.019531 |
| 3= | nomic-ai/nomic-embed-text-v1.5 | 0.007812 | 0.000000–0.019531 |
| 5= | BAAI/bge-large-en-v1.5 | 0.003906 | 0.000000–0.011719 |
| 5= | Snowflake/snowflake-arctic-embed-l | 0.003906 | 0.000000–0.011719 |
| 5= | intfloat/multilingual-e5-large | 0.003906 | 0.000000–0.011719 |
| 8 | BAAI/bge-m3 | 0.000000 | 0.000000–0.000000 |

**If `prec` were included in the headline** (`mean(proc, conc, time, prec)`,
keeping proc=conc=time=1.000), the resulting ranking would be:

| rank | agent | ARI-with-prec |
|---|---|---|
| 1 | sentence-transformers/all-MiniLM-L6-v2 | 0.773438 |
| 2 | sentence-transformers/all-mpnet-base-v2 | 0.758789 |
| 3= | mixedbread-ai/mxbai-embed-large-v1 | 0.751953 |
| 3= | nomic-ai/nomic-embed-text-v1.5 | 0.751953 |
| 5= | BAAI/bge-large-en-v1.5 | 0.750977 |
| 5= | Snowflake/snowflake-arctic-embed-l | 0.750977 |
| 5= | intfloat/multilingual-e5-large | 0.750977 |
| 8 | BAAI/bge-m3 | 0.750000 |

**But that numeric ranking is not as solid as it looks under CI.** Checking
pairwise overlap on the `prec` CIs (table 2b):

- MiniLM (0.0586–0.1289) **does not overlap** any other agent — it is
  clearly, statistically the worst under `prec`.
- mpnet (0.0156–0.0586) touches MiniLM's lower bound (0.0586) and **does
  not overlap** bge-large/arctic/e5-large (upper bound 0.0117) or bge-m3
  (0.0) — mpnet is distinguishable from those four. It does overlap
  mxbai/nomic (0–0.0195), so mpnet vs. mxbai/nomic is indistinguishable.
- mxbai and nomic are identical (same HER, same CI) — an exact tie, not
  just a numeric one.
- bge-large, arctic and e5-large are identical to each other (same HER,
  same CI) — an exact tie.
- The group {bge-large, arctic, e5-large, bge-m3, mxbai, nomic} overlaps
  internally in a chain (every CI includes 0 or comes very close), so
  **there is no reliable statistical separation inside that block of 6**,
  even though the numeric ranking splits them into 3 sub-groups.

Conclusion of this section: including `prec` does break the tie at 1.000
and does robustly identify one extreme (MiniLM and mpnet reproduce worse
under a precision change than the rest) — a counterintuitive result,
because those are exactly the two models `spec/ari-canonical-v0.1.md` §5
marks as `floor_limited` (no clean `κ`) in the synthetic drift-sensitivity
sweep. But the rest of the ranking, inside the block of 6 models with a
clean `κ`, is mostly point-estimate ordering without clear statistical
backing.

**A side note, outside the request but relevant to the "8 self-hosted"
frame:** `gemini/gemini-embedding-001` (class `api`) also ties at
ARI = 1.000 with the eight `self_hosted` agents, with all three core
conditions at exactly 1.000 and CI `[1.0, 1.0]`. The difference is that for
Gemini this is a *measurement* (no schema-enforced determinism control,
unlike `self_hosted`), and Gemini does not report `prec` — it is not
observable for an `api` agent (`spec/condition-set.md`,
"provider-internal"). There is no way to break the tie between Gemini and
the 8 self-hosted with the published data: there are 9 agents at 1.000, not
8, and that ninth lacks the diagnostic axis that would separate the other
eight.

## 5. Recommendation: derived views, not a redefinition of ARI

The published ARI has a versioned spec
([`spec/condition-set.md`](../../spec/condition-set.md),
[`spec/ari-canonical-v0.1.md`](../../spec/ari-canonical-v0.1.md)) and a
public [retraction ledger](../retractions.md) that has already withdrawn claims twice. This document is
not the place to propose a redefinition of the metric — the recommendation
is to add derived views computed **from data every submission already
publishes**, touching neither `ARI`, `spec/` nor the schema:

1. **A "Δ vs. same" column per core condition**, computed as
   `HER(k) − HER(same)`, shown next to the absolute HER in any per-agent
   detail view. It is direct arithmetic over fields that already exist in
   every JSON; no submitter recomputes anything.
2. **A per-cell classification badge** (real effect / indistinguishable /
   apparent improvement), using this document's CI non-overlap rule. This
   prevents a reader from reading "more sensitive to `time`" into an agent
   whose `time` is, in reality, indistinguishable from its own `same`
   noise (the OpenAI case and, partially, the Cohere case).
3. **An "ARI within `same` noise" badge** for `api` agents when the
   published ARI falls inside `same`'s CI, or when none of the three core
   components is a "real effect" — the OpenAI and Cohere cases in this
   dataset. It is a reading aid, not a score change.
4. **A separate diagnostic view explicitly labeled "non-canonical"**
   showing `mean(proc, conc, time, prec)` for agents that report `prec`
   (today, the 8 self-hosted), presented next to the official ARI and
   never in its place — this is literally what `spec/condition-set.md`
   already asks for ("Report them; read them alongside the headline, not
   inside it"), except that today that side-by-side reading exists in no
   public artifact. It must ship with section 4's CI-overlap breakdown:
   showing the ranking without its confidence bands would be more
   misleading than the current tie.
5. **Report the absence of `mach`/`lib` as coverage gaps**, not as zeros or
   "near 1.0" — today none of the 13 submissions includes them, and a
   derived view that omitted them silently would read as "no drift on that
   axis" instead of "no data".

None of these points changes any submission's `ARI` number or the
validation criteria of the board's scorer (`ari-leaderboard`'s `scoring/score.py`);
all are additional readings over data the submissions already publish.
